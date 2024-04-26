from flask import Flask, request, jsonify, render_template, url_for
from PIL import Image
import threading
import os
import sys
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
# Ensure the src directory and its submodules are accessible
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)
import src.spectral_utils as su
import src.graph_utils as gu
import src.localization as localization
import src.simgraph as sm

app = Flask(__name__, static_folder='static')
process_active = {}

# Define path to save processed images within the static directory
image_save_path = os.path.join(app.static_folder, 'processed_images')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

heatmap_path = os.path.join(image_save_path, 'heatmap_result.png')
# Delete existing file if it exists
if os.path.exists(heatmap_path):
    os.remove(heatmap_path)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global process_active
    thread_id = threading.get_ident()
    process_active[thread_id] = True

    try:
        while process_active[thread_id]:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            
            heatmap_path = os.path.join(image_save_path, 'heatmap_result.png')
            # Delete existing file if it exists
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)

            filename = 'normal-01'  # Set the standard filename
            extension = os.path.splitext(file.filename)[1]  # Get the file extension from the uploaded file
            filename += extension
            # Define the path to the testimages directory
            save_path = os.path.join(app.root_path, '..', 'testimages', filename)

            if os.path.exists(save_path):
                os.remove(save_path)

            file.save(save_path)
            
            # Image processing and analysis
            I2 = plt.imread(save_path)[:,:,:3]
            fweights = './models/cam_256/-30'
            patch_size = 256
            overlap = 0
            sg2 = sm.calc_simgraph(image=I2, f_weights_restore=fweights, patch_size=patch_size, overlap=overlap)
            M = gu.sym_mat(sg2.mat)
            g = gu.adj_to_graph(M, threshold=0.9)
            com, n, mod = gu.cluster_fastgreedy(g, n=2, weighted=True)
            overlap = 128 #patch sampling overlap
            sg2 = sm.calc_simgraph(image = I2, f_weights_restore = fweights, patch_size = patch_size, overlap = overlap)
            M = gu.sym_mat(sg2.mat) #symmetric similarity matrix for spliced image
            L = su.laplacian(M) #laplacian matrix
            gap = su.eigap01(L) #spectral gap
            print(f'Spectral Gap = {gap:.2f}')

            normL = su.laplacian(M, laplacian_type='sym') #normalized laplacian matrix
            normgap = su.eigap01(normL) #normalized spectral gap
            print(f'Normalized Spectral Gap = {normgap:.4f}')

            g = gu.adj_to_graph(M) #convert to igraph Graph object
            _, _, mod = gu.cluster_fastgreedy(g,weighted=True) #compute modularity
            print(f'Modularity Value = {mod:.4f}')
            
            fweights = './models/cam_128/-30' #path to model CNN weights
            patch_size = 128 #patch size, must match associated weights file
            overlap = 96 #patch sampling overlap

            sg2 = sm.calc_simgraph(image = I2, f_weights_restore = fweights, patch_size = patch_size, overlap = overlap)
            ## Forgery Localization - Modularity Optimization
            M = gu.sym_mat(sg2.mat) #symmetric similarity matrix for spliced image
            g = gu.adj_to_graph(M, threshold=0.7) #convert to igraph Graph object
            com, n, mod = gu.cluster_fastgreedy(g,n=2,weighted=True) #compute modularity

            pat_loc = localization.PatchLocalization(inds = sg2.inds, patch_size = patch_size,
                                                    prediction = com.membership)

            #f = pat_loc.plot_heatmap(image=I2)
            pix_loc = localization.pixel_loc_from_patch_pred(prediction=pat_loc.prediction,
                                                                inds = sg2.inds,
                                                                patch_size = patch_size,
                                                                image_shape = I2.shape[:2],
                                                                threshold = 0.5)
            L = su.laplacian(M)
            prediction = su.spectral_cluster(L)

            pat_loc = localization.PatchLocalization(inds = sg2.inds, patch_size = patch_size,
                                                    prediction = prediction)

            #f = pat_loc.plot_heatmap(image=I2,label=0)
            #here we flip the label for easier visualization..
            #note the label=0 in the line above (default=1)
            #and the ~pat_loc.prediction in the line below
            pix_loc = localization.pixel_loc_from_patch_pred(prediction=~pat_loc.prediction,
                                                            inds = sg2.inds,
                                                            patch_size = patch_size,
                                                            image_shape = I2.shape[:2],
                                                            threshold = 0.37)

            #pix_loc.plot(image=I2)

            ## Forgery Localization - Normalized Spectral Clustering
            L = su.laplacian(M,laplacian_type='sym')
            #normalization is performed with the laplacian_type parameter (default is None)
            #otherwise the analysis is the same

            prediction = su.spectral_cluster(L)

            pat_loc = localization.PatchLocalization(inds = sg2.inds, patch_size = patch_size,
                                                    prediction = ~prediction)
            #f = pat_loc.plot_heatmap(image=I2,label=0)
            #here we flip the label for easier visualization..
            #note the label=0 in the line above
            #and the ~pat_loc.prediction in the line below
            pix_loc = localization.pixel_loc_from_patch_pred(prediction=~pat_loc.prediction,
                                                            inds = sg2.inds,
                                                            patch_size = patch_size,
                                                            image_shape = I2.shape[:2],
                                                            threshold = 0.5)    
            #f = pix_loc.plot(image=I2)


            # Generating heatmap
            f = pat_loc.plot_heatmap(image=I2)
            heatmap_path = os.path.join(image_save_path, 'heatmap_result.png')

            # Delete existing file if it exists
            if os.path.exists(heatmap_path):
                os.remove(heatmap_path)

            plt.savefig(heatmap_path, bbox_inches='tight')
            
            # URL for the saved heatmap image
            # heatmap_url = url_for('static', filename=f'processed_images/heatmap_result.png')
            process_active[thread_id] = False
            pass
            return jsonify({'heatmap': f'/static/processed_images/heatmap_result.png'})
    finally:
        process_active[thread_id] = False
    
    return jsonify({'status': 'completed'})


@app.route('/cancel_process', methods=['POST'])
def cancel_process():
    global process_active
    thread_id = threading.get_ident()  # This would need to correspond to the process you want to stop
    process_active[thread_id] = False
    return jsonify({'status': 'cancelled'})

if __name__ == '__main__':
    app.run(debug=True)
