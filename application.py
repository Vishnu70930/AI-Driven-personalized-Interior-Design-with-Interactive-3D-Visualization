from flask import Flask, render_template, request, jsonify, redirect, send_from_directory
import os
import base64
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import vtkmodules.all as vtk
from vtkmodules.util import numpy_support
# import tkinter as tk
# from tkinter import filedialog
from flask import request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask import url_for
import uuid 
from vtkmodules.all import (
    vtkPLYReader, vtkPolyDataMapper, vtkActor,
    vtkRenderer, vtkRenderWindow, vtkRenderWindowInteractor,
    vtkLight, vtkNamedColors, vtkInteractorStyleTrackballCamera
)

# Load environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# Access the NVIDIA_API_KEY from the environment variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

if not NVIDIA_API_KEY:
    raise ValueError("No NVIDIA API Key found in environment variables.")

# Initialize Flask app
app = Flask(__name__)

app.secret_key = 'your_secret_key' 
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data['name']
    email = data['email']
    password = data['password']

    if User.query.filter_by(email=email).first():
        return jsonify({'status': 'fail', 'message': 'Email already exists'}), 409

    hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(name=name, email=email, password=hashed_pw)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'Signup successful'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data['email']
    password = data['password']

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        session['user_id'] = user.id
        return jsonify({'status': 'success', 'message': 'Login successful'})
    return jsonify({'status': 'fail', 'message': 'Invalid credentials'}), 401

# Set the path where images will be saved
IMAGE_FOLDER = os.path.join(os.getcwd(), 'static', 'generated_images')
UPLOAD_FOLDER = 'uploads'
PLY_FOLDER = os.path.join('static', 'ply_files')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLY_FOLDER, exist_ok=True)

def generate_image(style: str, color: str, room_width: float, room_length: float, room_type:str, other_params: str):
    
    invoke_url = "https://ai.api.nvidia.com/v1/genai/stabilityai/stable-diffusion-xl"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Accept": "application/json"}
    
    prompt = (
    f"Design a {room_type.replace('_', ' ')} with the following specifications:\n"
    f"- Interior Style: {style}\n"
    f"- Dominant Color Scheme: {color}\n"
    f"- Room Dimensions: {room_width} meters wide and {room_length} meters long\n"
    f"- Purpose and Ambiance: Create a {room_type.replace('_', ' ')} that reflects a {style} style with a {color} color scheme.\n"
    f"{f'- Additional Features: {other_params}' if other_params else '- No specific additional features'}"
)

    payload = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "cfg_scale": 5, "sampler": "K_DPM_2_ANCESTRAL", "seed": 0, "steps": 25
    }
    
    response = requests.post(invoke_url, headers=headers, json=payload)
    response.raise_for_status()
    response_body = response.json()
    
    if 'artifacts' in response_body:
        base64_string = response_body['artifacts'][0]['base64']
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        return image
    else:
        raise ValueError("No image found in response")

def image_to_ply_with_color(image_path, ply_output, mesh_style="realistic"):
    image = Image.open(image_path).convert('RGB')
    img_array = np.array(image)
    height, width, _ = img_array.shape

    x, y = np.meshgrid(np.linspace(0, width - 1, width),
                       np.linspace(0, height - 1, height))
    x = x.flatten()
    y = y.flatten()

    grayscale = np.mean(img_array, axis=2)

    if mesh_style == "lowpoly":
        reduction_factor = 10
        grayscale = grayscale[::reduction_factor, ::reduction_factor]
        img_array = img_array[::reduction_factor, ::reduction_factor]
        x, y = np.meshgrid(np.linspace(0, width - 1, grayscale.shape[1]),
                           np.linspace(0, height - 1, grayscale.shape[0]))
        x = x.flatten()
        y = y.flatten()

    # Use raw grayscale values directly (no normalization)
    z = grayscale.flatten()

    vertices_top = np.column_stack([x, y, z])
    vertices_bottom = np.column_stack([x, y, np.zeros_like(z)])
    vertices = np.vstack([vertices_top, vertices_bottom])

    faces = []
    rows, cols = grayscale.shape
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = i * cols + j + 1
            v2 = (i + 1) * cols + j
            v3 = (i + 1) * cols + j + 1
            if mesh_style == "realistic":
                faces.append([v0, v1, v2])
                faces.append([v1, v2, v3])
            elif mesh_style == "lowpoly":
                faces.append([v0, v1, v3])
                faces.append([v0, v2, v3])

    bottom_offset = len(vertices_top)
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = bottom_offset + i * cols + j
            v1 = bottom_offset + i * cols + j + 1
            v2 = bottom_offset + (i + 1) * cols + j
            v3 = bottom_offset + (i + 1) * cols + j + 1
            if mesh_style == "realistic":
                faces.append([v2, v1, v0])
                faces.append([v3, v2, v1])
            elif mesh_style == "lowpoly":
                faces.append([v3, v1, v0])
                faces.append([v3, v2, v0])

    for i in range(rows - 1):
        for j in range(cols - 1):
            if j == 0:
                v_top = i * cols + j
                v_bottom = bottom_offset + i * cols + j
                v_top_next = (i + 1) * cols + j
                v_bottom_next = bottom_offset + (i + 1) * cols + j
                faces.append([v_top, v_bottom, v_top_next])
                faces.append([v_bottom, v_bottom_next, v_top_next])
            if j == cols - 2:
                v_top = i * cols + j + 1
                v_bottom = bottom_offset + i * cols + j + 1
                v_top_next = (i + 1) * cols + j + 1
                v_bottom_next = bottom_offset + (i + 1) * cols + j + 1
                faces.append([v_top_next, v_bottom, v_top])
                faces.append([v_top_next, v_bottom_next, v_bottom])
            if i == 0:
                v_top = i * cols + j
                v_bottom = bottom_offset + i * cols + j
                v_top_next = i * cols + j + 1
                v_bottom_next = bottom_offset + i * cols + j + 1
                faces.append([v_top, v_bottom, v_top_next])
                faces.append([v_bottom, v_bottom_next, v_top_next])
            if i == rows - 2:
                v_top = (i + 1) * cols + j
                v_bottom = bottom_offset + (i + 1) * cols + j
                v_top_next = (i + 1) * cols + j + 1
                v_bottom_next = bottom_offset + (i + 1) * cols + j + 1
                faces.append([v_top_next, v_bottom, v_top])
                faces.append([v_top_next, v_bottom_next, v_bottom])

    colors = img_array.reshape(-1, 3)
    colors = np.vstack([colors, colors])

    with open(ply_output, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("comment Generated by Interior Design Generator\n")
        ply_file.write("end_header\n")

        for vertex, color in zip(vertices, colors):
            ply_file.write(f"{vertex[0]} {vertex[1]} {vertex[2]} {color[0]} {color[1]} {color[2]}\n")

        for face in faces:
            ply_file.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def display_ply_with_color(ply_file):

    colors = vtkNamedColors()
    renderer = vtkRenderer()
    renderer.SetBackground(colors.GetColor3d("SlateGray"))
    
    # Create a render window
    render_window = vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(1200, 800)
    render_window.SetWindowName("3D Interior Design Visualization")
    
    # Create an interactor
    interactor = vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Read the PLY file
    reader = vtkPLYReader()
    reader.SetFileName(ply_file)
    
    # Create a mapper
    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    
    # Create an actor with enhanced properties
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.8)
    actor.GetProperty().SetSpecular(0.9)
    actor.GetProperty().SetSpecularPower(50)
    
    # Add the actor to the scene
    renderer.AddActor(actor)
    
    # Add lights for better visualization
    light1 = vtkLight()
    light1.SetPosition(0, 0, 1000)
    light1.SetIntensity(0.7)
    renderer.AddLight(light1)
    
    light2 = vtkLight()
    light2.SetPosition(1000, 1000, 1000)
    light2.SetIntensity(0.5)
    renderer.AddLight(light2)
    
    light3 = vtkLight()
    light3.SetPosition(-1000, -1000, 500)
    light3.SetIntensity(0.3)
    renderer.AddLight(light3)
    
    # Add a camera with a nice starting position
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, -500, 300)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCamera()
    
    # Set up interaction style
    style = vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)
    
    # Start the visualization
    render_window.Render()
    interactor.Start()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html") 

@app.route("/design")
def design_page():
    return render_template("design.html")

@app.route("/generate", methods=["POST"])
def generate():
    # Receive form data from frontend
    style = request.form.get("style")
    color = request.form.get("color")
    room_width = float(request.form.get("room_width", 0))
    room_length = float(request.form.get("room_length", 0))
    room_type = request.form.get("room_type")
    other_params = request.form.get("other_params")
    
    try:
        image = generate_image(style, color, room_width, room_length, room_type, other_params)
        
        # Save the generated image with a unique filename
        unique_id = str(uuid.uuid4())
        image_filename = f"generated_image_{unique_id}.png"
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        image.save(image_path)

        # Provide the URL for the user to view the image
        image_url = f'/static/generated_images/{image_filename}'
        
        return jsonify({
            "status": "success", 
            "image_url": image_url,
            "image_id": unique_id  
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/convert_to_3d/<image_id>")
def convert_to_3d(image_id):
    try:
        image_filename = f"generated_image_{image_id}.png"
        image_path = os.path.join(IMAGE_FOLDER, image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({"status": "error", "message": "Image not found"}), 404

        # Changed to save in ply_files folder
        ply_output = os.path.join(PLY_FOLDER, f"model_{image_id}.ply")
        image_to_ply_with_color(image_path, ply_output)
        
        display_ply_with_color(ply_output)
        
        return jsonify({
            "status": "success",
            "message": "3D model generated and displayed",
            "ply_path": f'/static/ply_files/model_{image_id}.ply'  # Updated path
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if 'image' not in request.files:
            return redirect('/')
        
        file = request.files['image']
        if file.filename == '':
            return redirect('/')
        
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            unique_id = str(uuid.uuid4())
            ply_output = os.path.join(PLY_FOLDER, f"model_{unique_id}.ply")
            image_to_ply_with_color(filepath, ply_output)
            
            # Display the PLY file
            display_ply_with_color(ply_output)
            
            return redirect(url_for('gallery'))
    
    return '''
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="image" required>
        <input type="submit" value="Upload">
    </form>
    '''
@app.route('/gallery')
def gallery():
    image_folder = os.path.join(app.static_folder, 'generated_images')
    image_list = os.listdir(image_folder)
    print("Images found:", image_list) 
    return render_template('gallery.html', images=image_list)

@app.route("/view3d/<filename>")
def view3d(filename):
    try:
        image_path = os.path.join(IMAGE_FOLDER, filename)
        if not os.path.exists(image_path):
            return "Image not found.", 404

        ply_output = os.path.join(PLY_FOLDER, f"model_{os.path.splitext(filename)[0]}.ply")
        image_to_ply_with_color(image_path, ply_output)
        display_ply_with_color(ply_output)
        return redirect(url_for('gallery'))

    except Exception as e:
        return f"Error rendering 3D model: {str(e)}"

@app.route("/download/<filename>")
def download_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
