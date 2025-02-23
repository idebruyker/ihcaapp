from flask import Flask, render_template, request, send_file
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import StringIO
import io
import numpy as np
import base64
import warnings

app = Flask(__name__)

# general
warnings.filterwarnings("ignore") # only warnings show for xticlables and yticklabels, but have not effect except for filling up terminal window

# Load the saved model
model_filename = 'best_model.pkl'

if os.path.isfile(model_filename):
    print(f'Loading model from: {model_filename}')
else:
    print(f'Model file not found: {model_filename}')

loaded_model = joblib.load(model_filename)

# # Configure upload and processed folders
# UPLOAD_FOLDER = 'uploads'
# PROCESSED_FOLDER = 'processed'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# title
sample_title = 'Immuno Histo Chemistry Analysis'
            
# # Ensure folders exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(PROCESSED_FOLDER, exist_ok=True)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            # read file
            file_content = file.read() #read file content as bytes
            text = file_content.decode('utf-8') #decode bytes to string
            # Process the CSV file
            df = pd.read_csv(StringIO(text),usecols=[7,8,21,27,33,39,51]) #read string as csv
            df.rename(columns={'Centroid X µm': 'Centroid.X.µm', 
                        'Centroid Y µm': 'Centroid.Y.µm', 
                        'Nucleus: Opal 480 mean': 'Nucleus..Opal.480.mean', 
                        'Nucleus: Opal 520 mean': 'Nucleus..Opal.520.mean', 
                        'Nucleus: Opal 570 mean': 'Nucleus..Opal.570.mean', 
                        'Nucleus: Opal 620 mean': 'Nucleus..Opal.620.mean', 
                        'Nucleus: Opal 690 mean': 'Nucleus..Opal.690.mean'}, 
                        inplace=True)
            
            # predict
            predictions = loaded_model.predict(df)
            # Add predictions to the new dataset 
            df['Predictions'] = predictions

            # present
            # types 
            # 10 = CD8
            # 20 = CD4
            # 30 = mhcII
            # 40 = pd1
            # 50 = pd1tcf
            # 60 = tcf

            # slices
            filtered_10 = df[(df['Predictions'] == 10)]
            filtered_20 = df[(df['Predictions'] == 20)]
            filtered_30 = df[(df['Predictions'] == 30)]
            filtered_40 = df[(df['Predictions'] == 40)]
            filtered_50 = df[(df['Predictions'] == 50)]
            filtered_60 = df[(df['Predictions'] == 60)]

            # dfsets
            df_plot1 = pd.concat([filtered_10,filtered_40,filtered_50,filtered_60])
            df_plot1['Predictions'] = df_plot1['Predictions'].replace(40, 10) #replace pd1 with cd8
            df_plot1['Predictions'] = df_plot1['Predictions'].replace(50, 10) #replace pd1tcf with cd8
            df_plot1['Predictions'] = df_plot1['Predictions'].replace(60, 10) #replace tcf with cd8

            df_plot2 = pd.concat([filtered_30])

            # Create a dictionary to map types to colors
            color_map_plot1 = {10: 'lightgray', 40: 'red', 50: '#39FF14'}
            color_map_plot2 = {30: 'gray'}
            color_map_plot3 = {30: 'blue'}

            colors_plot1 = df_plot1['Predictions'].map(color_map_plot1)
            colors_plot2 = df_plot2['Predictions'].map(color_map_plot2)

            plt.switch_backend('Agg')  # Switch to 'Agg' to prevent the figure from being displayed

            # create grid
            fig = plt.figure(layout=None, figsize=(30,11)) 
            # gs = fig.add_gridspec(nrows=3, ncols=3, hspace=0.2, wspace=0.2, left=0.05, right=0.95, top=0.95, bottom=0.05)
            gs = gridspec.GridSpec(3,3, width_ratios=[1,1,1], height_ratios=[6,1,1]) # 3 rows, 3 columns
            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])
            ax3 = fig.add_subplot(gs[1, :])
            ax4 = fig.add_subplot(gs[2, :])

            ############### plot 1 ####################################################################################################################
            ax0.scatter(df_plot1['Centroid.X.µm'], df_plot1['Centroid.Y.µm']*(-1), s=4/100, c=colors_plot1, marker='.')
            ax0.set_xlabel("Centroid X µm", fontsize=10)
            ax0.set_xticklabels(ax0.get_xticklabels(), fontsize=4, va='center')
            ax0.set_ylabel("Centroid Y µm", fontsize=10)
            ax0.set_yticklabels(ax0.get_yticklabels(), rotation=90, fontsize=4, va='center')
            ax0.legend(handles=[plt.Line2D([], [], color='gray', marker='o', label='CD8+ Cells'),
                                plt.Line2D([], [], color='red', marker='o', label='CD8+PD1+TCF- Cells'),
                                plt.Line2D([], [], color='#39FF14', marker='o', label='CD8+PD1+TCF+ Cells')],
                                fontsize=8)
            ax0.set_box_aspect(1)
            ############### plot 2 ####################################################################################################################
            ax1.scatter(df_plot2['Centroid.X.µm'], df_plot2['Centroid.Y.µm']*(-1), s=4/100,c=colors_plot2, marker='.')
            ax1.set_xlabel("Centroid X µm", fontsize=10)
            ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=4, va='center')
            ax1.set_ylabel("Centroid Y µm", fontsize=10)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=90, fontsize=4, va='center')
            ax1.legend(handles=[plt.Line2D([], [], color='gray', marker='o', label='MHC II+')], fontsize=8)
            ax1.set_box_aspect(1)
            ############### plot 3 ####################################################################################################################
                    # only mhcII and pd1tcf
            df_in = pd.concat([filtered_30, filtered_50])

            # delete columns
            df_in.drop('Nucleus..Opal.570.mean', axis=1, inplace=True)
            df_in.drop('Nucleus..Opal.690.mean', axis=1, inplace=True)
            df_in.drop('Nucleus..Opal.480.mean', axis=1, inplace=True)
            df_in.drop('Nucleus..Opal.620.mean', axis=1, inplace=True)
            df_in.drop('Nucleus..Opal.520.mean', axis=1, inplace=True)

            # round to nearest 100 nanometers -> mm
            df_in['Centroid.X.µm'] = df_in['Centroid.X.µm'] // 100 * 1 #100
            df_in['Centroid.Y.µm'] = df_in['Centroid.Y.µm'] // 100 * 1 #100

            # max / min X and Y
            max_x = df_in['Centroid.X.µm'].max()
            min_x = df_in['Centroid.X.µm'].min()
            step_x = 100
            max_y = df_in['Centroid.Y.µm'].max()
            min_y = df_in['Centroid.Y.µm'].min()
            step_y = 100

            # generate x and y values
            x_values = np.arange(min_x, max_x, 1) #100
            y_values = np.arange(min_y, max_y, 1) #100

            # split df into mhcii and pd1tcf
            df_mhcii = df_in[df_in['Predictions'] == 30].copy()
            df_pd1tcf = df_in[df_in['Predictions'] == 50].copy()

            # Merge the two dfFrames on 'x' and 'y'
            df_plot3 = pd.merge(df_mhcii, df_pd1tcf, on=['Centroid.X.µm', 'Centroid.Y.µm'], how='inner')
            
            # create plot
            ax2.scatter(df_plot3['Centroid.X.µm'], df_plot3['Centroid.Y.µm']*(-1), s=4/100, marker='.') #c=colors_plot3
            ax2.set_xlabel("Centroid X mm", fontsize=10)
            ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=4, va='center')
            ax2.set_ylabel("Centroid Y mm", fontsize=10)
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=90, fontsize=4, va='center')
            ax2.legend(handles=[plt.Line2D([], [], color='blue', marker='o', label='Immune Niches')], fontsize=8)
            ax2.set_box_aspect(1)
            ############### plot 4 ####################################################################################################################
            # total
            total_max_x = df['Centroid.X.µm'].max() 
            total_min_x = df['Centroid.X.µm'].min()
            total_max_y = df['Centroid.Y.µm'].max() 
            total_min_y = df['Centroid.Y.µm'].min() 
            total_area = ((total_max_x - total_min_x) * (total_max_y - total_min_y))  // 10000  # convert to mm^2
            total_count = df.shape[0]
            
            # all cells
            df_all = df.copy()
            # delete columns
            df_all.drop('Nucleus..Opal.570.mean', axis=1, inplace=True)
            df_all.drop('Nucleus..Opal.690.mean', axis=1, inplace=True)
            df_all.drop('Nucleus..Opal.480.mean', axis=1, inplace=True)
            df_all.drop('Nucleus..Opal.620.mean', axis=1, inplace=True)
            df_all.drop('Nucleus..Opal.520.mean', axis=1, inplace=True)

            # round to nearest 100 nanometers -> mm
            df_all['Centroid.X.µm'] = df_all['Centroid.X.µm'] // 100 * 1 #100
            df_all['Centroid.Y.µm'] = df_all['Centroid.Y.µm'] // 100 * 1 #100
            
            # delete duplicates
            df_all = df_all.drop_duplicates()
            single_cells_total = df_all.shape[0]
        
            #  cd8 = 10
            df_cd8 = pd.concat([df[df['Predictions'] == 10],df[df['Predictions'] == 40],df[df['Predictions'] == 50],df[df['Predictions'] == 60]],ignore_index=True)
            cd8_max_x = df['Centroid.X.µm'].max() 
            cd8_min_x = df['Centroid.X.µm'].min()
            cd8_max_y = df['Centroid.Y.µm'].max() 
            cd8_min_y = df['Centroid.Y.µm'].min() 
            cd8_area = ((cd8_max_x - cd8_min_x) * (cd8_max_y - cd8_min_y))  // 10000  # convert to mm^2
            cd8_count = df_cd8.shape[0]
            cd8_mm2 = cd8_count / cd8_area
            cd8_DAPI = cd8_count / total_count

            #  mhcII = 30
            df_mhcII = df[df['Predictions'] == 30]
            mhcII_max_x = df['Centroid.X.µm'].max() 
            mhcII_min_x = df['Centroid.X.µm'].min()
            mhcII_max_y = df['Centroid.Y.µm'].max() 
            mhcII_min_y = df['Centroid.Y.µm'].min() 
            mhcII_area = ((mhcII_max_x - mhcII_min_x) * (mhcII_max_y - mhcII_min_y))  // 10000  # convert to mm^2
            mhcII_count = df_mhcII.shape[0]
            mhcII_mm2 = mhcII_count / mhcII_area
            mhcII_DAPI = mhcII_count / total_count

            #  cd4 = 20
            df_cd4 = df[df['Predictions'] == 20]
            cd4_max_x = df['Centroid.X.µm'].max() 
            cd4_min_x = df['Centroid.X.µm'].min()
            cd4_max_y = df['Centroid.Y.µm'].max() 
            cd4_min_y = df['Centroid.Y.µm'].min() 
            cd4_area = ((cd4_max_x - cd4_min_x) * (cd4_max_y - cd4_min_y))  // 10000  # convert to mm^2
            cd4_count = df_cd4.shape[0]
            cd4_mm2 = cd4_count / cd4_area
            cd4_DAPI = cd4_count / total_count

            #  tcf = 60,50
            df_tcf = pd.concat([df[df['Predictions'] == 60],df[df['Predictions'] == 50]],ignore_index=True)
            tcf_max_x = df['Centroid.X.µm'].max() 
            tcf_min_x = df['Centroid.X.µm'].min()
            tcf_max_y = df['Centroid.Y.µm'].max() 
            tcf_min_y = df['Centroid.Y.µm'].min() 
            tcf_area = ((tcf_max_x - tcf_min_x) * (tcf_max_y - tcf_min_y))  // 10000  # convert to mm^2
            tcf_count = df_tcf.shape[0]
            tcf_mm2 = tcf_count / tcf_area
            tcf_DAPI = tcf_count / total_count

            #  tcfneg = 40
            df_tcfneg = df[df['Predictions'] == 40] 
            tcfneg_max_x = df['Centroid.X.µm'].max() 
            tcfneg_min_x = df['Centroid.X.µm'].min()
            tcfneg_max_y = df['Centroid.Y.µm'].max() 
            tcfneg_min_y = df['Centroid.Y.µm'].min() 
            tcfneg_area = ((tcfneg_max_x - tcfneg_min_x) * (tcfneg_max_y - tcfneg_min_y))  // 10000  # convert to mm^2
            tcfneg_count = df_tcfneg.shape[0]
            tcfneg_mm2 = tcfneg_count / tcfneg_area
            tcfneg_DAPI = tcfneg_count / total_count

            #  pd1 = 40,50
            df_pd1 = pd.concat([df[df['Predictions'] == 40],df[df['Predictions'] == 50]],ignore_index=True)
            pd1_max_x = df['Centroid.X.µm'].max() 
            pd1_min_x = df['Centroid.X.µm'].min()
            pd1_max_y = df['Centroid.Y.µm'].max() 
            pd1_min_y = df['Centroid.Y.µm'].min() 
            pd1_area = ((pd1_max_x - pd1_min_x) * (pd1_max_y - pd1_min_y))  // 10000  # convert to mm^2
            pd1_count = df_pd1.shape[0]
            pd1_mm2 = pd1_count / pd1_area
            pd1_DAPI = pd1_count / total_count

            #  pd1tcf = 50
            df_pd1tcf = df[df['Predictions'] == 50]
            pd1tcf_max_x = df['Centroid.X.µm'].max() 
            pd1tcf_min_x = df['Centroid.X.µm'].min()
            pd1tcf_max_y = df['Centroid.Y.µm'].max() 
            pd1tcf_min_y = df['Centroid.Y.µm'].min() 
            pd1tcf_area = ((pd1tcf_max_x - pd1tcf_min_x) * (pd1tcf_max_y - pd1tcf_min_y))  // 10000  # convert to mm^2
            pd1tcf_count = df_pd1tcf.shape[0]
            pd1tcf_mm2 = pd1tcf_count / pd1tcf_area
            pd1tcf_DAPI = pd1tcf_count / total_count

            # niche proportion
            df_plot3 = df_plot3.drop_duplicates()
            single_cells_niche = df_plot3.shape[0]
            nicheproportion = single_cells_niche / single_cells_total
            nicheproportion_percentage = nicheproportion * 100

            # build numbers table to download
            numbers_table = pd.DataFrame({
                'Sample': [sample_title],
                'CD8 area': [cd8_area],
                'CD8 count': [cd8_count],
                'CD8 per mm2': [cd8_mm2],
                'CD8 per DAPI': [cd8_DAPI],
                'MHCII area': [mhcII_area],
                'MHCII count': [mhcII_count],
                'MHCII per mm2': [mhcII_mm2],
                'MHCII per DAPI': [mhcII_DAPI],
                'CD4 area': [cd4_area],
                'CD4 count': [cd4_count],
                'CD4 per mm2': [cd4_mm2],
                'CD4 per DAPI': [cd4_DAPI],
                'TCF area': [tcf_area],
                'TCF count': [tcf_count],
                'TCF per mm2': [tcf_mm2],
                'TCF per DAPI': [tcf_DAPI],
                'TCFneg area': [tcfneg_area],
                'TCFneg count': [tcfneg_count],
                'TCFneg per mm2': [tcfneg_mm2],
                'TCFneg per DAPI': [tcfneg_DAPI],
                'PD1 area': [pd1_area],
                'PD1 count': [pd1_count],
                'PD1 per mm2': [pd1_mm2],
                'PD1 per DAPI': [pd1_DAPI],
                'PD1TCF area': [pd1tcf_area],
                'PD1TCF count': [pd1tcf_count],
                'PD1TCF per mm2': [pd1tcf_mm2],
                'PDTCF per DAPI': [pd1tcf_DAPI],
                'Niche Proportion': [nicheproportion],
                'Niche Percent': [nicheproportion_percentage]})
            
            # build numbers table to display
            numbers_table_ax3 = numbers_table[numbers_table.columns[1:17]]
            numbers_table_ax4 = numbers_table[numbers_table.columns[17:34]] #up by 2 once niche fields added

            displaytable_ax3 = ax3.table(cellText=numbers_table_ax3.values, colLabels=numbers_table_ax3.columns, cellLoc = 'center', colWidths=[0.05]*len(numbers_table_ax3.columns), loc='center')
            displaytable_ax3.set_fontsize(10)  #not working as autosize
            # displaytable.

            ax3.axis('off')

            ax3.set_title('Numbers/Output df', fontsize=10,color='red', loc='center')
            
            displaytable_ax4 = ax4.table(cellText=numbers_table_ax4.values, colLabels=numbers_table_ax4.columns, cellLoc = 'center', colWidths=[0.05]*len(numbers_table_ax4.columns), loc='center')
            displaytable_ax4.set_fontsize(10)  #not working as autosize
            # displaytable.

            ax4.axis('off')
            ############### general and show ####################################################################################################################
            fig.suptitle(sample_title, fontsize=20, color='red', fontweight='bold')  # Add title to the figure

            # # Save the plot to BytesIO object
            # buf = io.BytesIO()
            # plt.savefig(buf, dpi=300, bbox_inches='tight', format='png') #1200 possible
            # plt.close()
            # buf.seek(0)

            # # Encode the image in Base64
            # plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Example: Generate a plot
            # plt.figure()
            # df.plot(kind='bar')  # Example plot (customize as needed)

            # plt.savefig(plot_path)

            # Save the plot
            plot_png = f"IHCA.png"
            plot_path = os.path.join('static', plot_png)
            # plot_path = os.path.join(app.config['PROCESSED_FOLDER'], plot_png)
            print(plot_path)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight') #1200 possible
            # plt.show()
            plt.close()


            # Provide download links
            return render_template('index.html', 
                                    plot_url=plot_path)
                                    # plot_url=plot_data)
                                #  plot_url=plot_path, 
                                #  csv_url=processed_csv_path)

    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join('static', filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)