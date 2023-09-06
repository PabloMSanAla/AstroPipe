from PyQt5 import QtWidgets, uic
from ScientificDoubleSpinBox import ScientificDoubleSpinBox
import sys
import json
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import pickle as pkl

class ui(QtWidgets.QMainWindow):
    def __init__(self):
        super(ui, self).__init__()
        uic.loadUi('main.ui', self)

        # Initialization flag
        self.initialization = True

        # --------------------------------------------------------------------------------------------------------------
        # INITIALIZATION OF MENU ITEMS
        # --------------------------------------------------------------------------------------------------------------
        # Link the save menu item to the save_imp_file method
        self.generate_imp = self.findChild(QtWidgets.QAction, 'actionGenerate_IMP')
        self.generate_imp.triggered.connect(self.generate_imp_file)
        self.load_data = self.findChild(QtWidgets.QAction, 'actionLoad_data')
        self.load_data.triggered.connect(self.load_imp_data)
        self.load_defaults = self.findChild(QtWidgets.QAction, 'actionClear')
        self.load_defaults.triggered.connect(self.load_default_data)
        self.save_data = self.findChild(QtWidgets.QAction, 'actionSave_data')
        self.save_data.triggered.connect(self.save_imp_file)
        self.close_app = self.findChild(QtWidgets.QAction, 'actionClose')
        self.close_app.triggered.connect(self.close_gui)
        self.appearance = self.findChild(QtWidgets.QAction, 'actionAppearance')
        self.appearance.triggered.connect(self.change_gui_appearance)
        self.about = self.findChild(QtWidgets.QAction, 'actionAbout')
        self.about.triggered.connect(self.show_about)
        self.load_fits = self.findChild(QtWidgets.QAction, 'actionLoad_observations_fits')
        self.load_fits.triggered.connect(self.load_fits_data)
        # --------------------------------------------------------------------------------------------------------------
        # MAIN DISPLAY CANVAS
        # --------------------------------------------------------------------------------------------------------------
        self.main_display = self.findChild(QtWidgets.QGroupBox, 'main_display')

        # create the figure to show 4 images as subplots
        self.display_canvas_figure, self.display_canvas_ax = plt.subplots(2,2, figsize=(1,3), 
                                                                          sharex=True, sharey=True,
                                                                          constrained_layout=True)
        # create a canvas, add it to the layout and set the layout back to the frame layout
        self.display_canvas = FigureCanvas(self.display_canvas_figure)
        layout = self.main_display.layout()
        layout.addWidget(self.display_canvas)
        self.main_display.setLayout(layout)     
        # --------------------------------------------------------------------------------------------------------------
        # DATA
        # --------------------------------------------------------------------------------------------------------------
        self.wavelength_selector = self.findChild(QtWidgets.QSlider, 'wavelength_selector')
        self.number_wavelengths = self.findChild(QtWidgets.QLabel, 'number_wavelengths')
        self.displayed_wavelength_label = self.findChild(QtWidgets.QLabel, 'displayed_wavelength')
        self.x_resolution = self.findChild(QtWidgets.QLabel, 'x_resolution')
        self.y_resolution = self.findChild(QtWidgets.QLabel, 'y_resolution')
        self.fov_x = self.findChild(QtWidgets.QDoubleSpinBox, 'fov_x')
        self.fov_y = self.findChild(QtWidgets.QDoubleSpinBox, 'fov_y')
        self.instrument = self.findChild(QtWidgets.QComboBox, 'instrument_selector')

        # connect the wavelength selector to the update_displayed_wavelength method
        self.wavelength_selector.valueChanged.connect(lambda: self.update_displayed_wavelength(self.initialization))
        # --------------------------------------------------------------------------------------------------------------
        # STOKES
        # --------------------------------------------------------------------------------------------------------------
        # retrieve the number wavelengths QspinBox
        self.stokes_canvas = self.findChild(QtWidgets.QFrame, 'stokes_canvas')
        self.I_weight = self.findChild(QtWidgets.QDoubleSpinBox, 'I_weight')
        self.Q_weight = self.findChild(QtWidgets.QDoubleSpinBox, 'Q_weight')
        self.U_weight = self.findChild(QtWidgets.QDoubleSpinBox, 'U_weight')
        self.V_weight = self.findChild(QtWidgets.QDoubleSpinBox, 'V_weight')
        self.noise_level = self.findChild(QtWidgets.QDoubleSpinBox, 'noise_level')
        self.noise_type = self.findChild(QtWidgets.QComboBox, 'noise_type')
        self.select_spectra_button = self.findChild(QtWidgets.QPushButton, 'select_spectra_button')

        # create a figure to plot on
        self.stokes_figure = plt.figure(figsize=(3,2), constrained_layout=True)
        # set the axes to the figure to be from 0 to 1 and adding to the figure
        ax = plt.Axes(self.stokes_figure, [0., 0., 1., 1.])
        self.stokes_figure.add_axes(ax)
        # create a canvas, add it to the layout and set the layout back to the frame layout
        self.stokes_plot_canvas = FigureCanvas(self.stokes_figure)
        layout = self.stokes_canvas.layout()
        layout.addWidget(self.stokes_plot_canvas)
        self.stokes_canvas.setLayout(layout)
        # --------------------------------------------------------------------------------------------------------------
        # GEOMETRY
        # --------------------------------------------------------------------------------------------------------------
        # retrieve the button and canvas from the UI and connect the button to the function
        self.capture_coordinates = self.findChild(QtWidgets.QPushButton, 'capture_coordinates')
        self.geometry_canvas = self.findChild(QtWidgets.QFrame, 'geometry_canvas')
        self.mask_coordinates = self.findChild(QtWidgets.QLineEdit, 'mask_coordinates')
        self.save_mask = self.findChild(QtWidgets.QPushButton, 'save_mask')
        self.capture_origin = self.findChild(QtWidgets.QPushButton, 'capture_origin')
        # Observer vector and orientation
        self.observer_x = self.findChild(QtWidgets.QDoubleSpinBox, 'observer_x')
        self.observer_y = self.findChild(QtWidgets.QDoubleSpinBox, 'observer_y')
        self.observer_z = self.findChild(QtWidgets.QDoubleSpinBox, 'observer_z')
        self.observer_phi = self.findChild(QtWidgets.QDoubleSpinBox, 'observer_phi')
        self.observer_theta = self.findChild(QtWidgets.QDoubleSpinBox, 'observer_theta')
        self.observer_psi = self.findChild(QtWidgets.QDoubleSpinBox, 'observer_psi')
        # domain dimensions, vector and orientation
        self.domain_phi = self.findChild(QtWidgets.QDoubleSpinBox, 'domain_phi')
        self.domain_theta = self.findChild(QtWidgets.QDoubleSpinBox, 'domain_theta')
        self.domain_psi = self.findChild(QtWidgets.QDoubleSpinBox, 'domain_psi')
        self.domain_x = self.findChild(QtWidgets.QDoubleSpinBox, 'domain_x')
        self.domain_y = self.findChild(QtWidgets.QDoubleSpinBox, 'domain_y')
        self.domain_z = self.findChild(QtWidgets.QDoubleSpinBox, 'domain_z')
        self.dimension_x = self.findChild(QtWidgets.QDoubleSpinBox, 'dimension_x')
        self.dimension_y = self.findChild(QtWidgets.QDoubleSpinBox, 'dimension_y')
        self.dimension_z = self.findChild(QtWidgets.QDoubleSpinBox, 'dimension_z')

        # initialize the list of coordinates
        self.geometry_coordinates = []

        # find the Qframe layout
        layout = self.geometry_canvas.layout()

        # create a figure to plot on
        self.geometry_figure = plt.figure()

        # set the axes to the figure to be from 0 to 1 and adding to the figure
        ax = plt.Axes(self.geometry_figure, [0., 0., 1., 1.])
        self.geometry_figure.add_axes(ax)

        # create a canvas, add it to the layout and set the layout back to the frame layout
        self.geometry_plot_canvas = FigureCanvas(self.geometry_figure)
        layout.addWidget(self.geometry_plot_canvas)
        self.geometry_canvas.setLayout(layout)
        # actually draw the figure on the canvas with just the background image
        # self.plot_polygon(self.geometry_coordinates)

        # connect the button to the function to capture the coordinates and save the coordinates
        self.capture_coordinates.clicked.connect(self.capture_geometry_coordinates)
        self.mask_coordinates.textChanged.connect(self.set_mask_coordinates)
        self.save_mask.clicked.connect(self.save_mask_coordinates)

        # --------------------------------------------------------------------------------------------------------------
        # ATOM
        # --------------------------------------------------------------------------------------------------------------
        self.atomic_canvas = self.findChild(QtWidgets.QFrame, 'atomic_canvas')
        self.atomic_module = self.findChild(QtWidgets.QComboBox, 'atomic_module')

        # --------------------------------------------------------------------------------------------------------------
        # QUANTITIES
        # --------------------------------------------------------------------------------------------------------------
        self.quantities_canvas = self.findChild(QtWidgets.QFrame, 'quantities_canvas')
        self.quant0 = self.findChild(QtWidgets.QCheckBox, 'quant0')
        # self.quant1 = self.findChild(QtWidgets.QComboBox, 'quant1')

        # --------------------------------------------------------------------------------------------------------------
        # ALGORITHMS
        # --------------------------------------------------------------------------------------------------------------
        self.algorithms_canvas = self.findChild(QtWidgets.QFrame, 'algorithm_canvas')
        self.algorithm = self.findChild(QtWidgets.QComboBox, 'algorithm_selector')


        # --------------------------------------------------------------------------------------------------------------
        self.load_default_data()        
        self.show()

    # --------------------------------------------------------------------------------------------------------------
    # MENU ITEMS METHODS
    # --------------------------------------------------------------------------------------------------------------
    def load_fits_data(self):
        """
        Loads the fits data from the file
        """
        # get the file name from the user
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '', 'Pickle files (*.pkl)')[0]
        # if the file name is not empty
        if file_name != '':
            # load the fits data with astropy
            # fits_data = fits.open(file_name)
            with open(file_name, "rb") as input_file:
                fits_data = pkl.load(input_file)

            # get the data from the fits data
            self.stokes_data = fits_data[0]['data']
            self.x_resolution.setText(str(fits_data[0]['data'].shape[1]))
            self.y_resolution.setText(str(fits_data[0]['data'].shape[2]))
            self.number_wavelengths.setText(str(fits_data[0]['data'].shape[3]))
            # get the wavelength from the fits data
            self.frequency_array = fits_data[0]['lambda']
            self.wavelength_selector.setMaximum(self.frequency_array.shape[0]-1)
        else:
            print('No file selected')
        
        # update the plots
        self.update_stokes_preview()
        self.plot_polygon(self.geometry_coordinates)
        self.update_main_display()


    def load_imp_data(self):
        # This is executed when the save menu item is pressed
        print('Selecting file to load...')
        # open a file dialog to select the file
        file_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '.', 'IMP files (*.imp)')
        if file_name[0] != '':
            # load the file
            print('Loading an IMP file...')
            with open(file_name[0], 'r') as infile:
                self.input_data = json.load(infile)
            print('IMP file loaded!')
            # update the UI
            try:
                self.update_ui()
            except Exception as e:
                # open a dialog Box explaining the error loading the file
                print('Error loading file')
                self.error_dialog = QtWidgets.QMessageBox()
                self.error_dialog.setWindowTitle('Error')
                self.error_dialog.setIcon(QtWidgets.QMessageBox.Critical)
                self.error_dialog.setText('Error loading file')
                self.error_dialog.setInformativeText(str(e))
                self.error_dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
                self.error_dialog.exec_()


    def load_default_data(self):
        # This is executed when the save menu item is pressed
        print('Loading default IMP file...')
        self.input_data = json.load(open('default.imp'))
        print('Default IMP file loaded!')
        # update the UI
        self.update_ui()
        self.initialization = False

    def generate_imp_file(self):
        # This is executed when the generate IMP menu item is pressed
        print('Generating an IMP file...')
        self.output_data = self.get_gui_data()
        # write the data to console
        # print(json.dumps(self.output_data, indent=2))
    
    def save_imp_file(self):
        # This is executed when the save menu item is pressed
        # open a file dialog to select where to store the file
        self.save_file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save file', '.', 'IMP files (*.imp)')
        self.save_file_name = self.save_file_name[0]
        # if the file name is not empty then save the file (if it is empty then the user cancelled the save)
        if self.save_file_name != '':
            # save the file
            # generate the imp and print it to the file
            self.generate_imp_file()
            print('Saving an IMP file...')
            with open(self.save_file_name, 'w') as outfile:
                json.dump(self.output_data, outfile, indent=4)
            print('IMP file saved!')

    def close_gui(self):
        # This is executed when the close menu item is pressed
        print('Closing the application...')        
        # ask the user if they want to save the file
        reply = QtWidgets.QMessageBox.question(self, 'Save file?', 'Do you want to save the file?',
                                                QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        # if the user wants to save the file then save it
        if reply == QtWidgets.QMessageBox.Yes:
            self.save_imp_file()
        # close the application
        self.close()

    def change_gui_appearance(self):
        # This is executed when the change gui appearance menu item is pressed
        print('Changing the appearance of the gui...')
        # open a file dialog to select the new style from the default styles
        print(QtWidgets.QStyleFactory.keys())
        # self.setStyle(QtWidgets.QStyleFactory.create('Windows'))

    def show_about(self):
        # This is executed when the about menu item is pressed
        print('Showing about dialog...')
        # show the about dialog
        QtWidgets.QMessageBox.about(self, 'About IMP', 'IMP is a tool for generating and analyzing IMP files.')

    # --------------------------------------------------------------------------------------------------------------
    # MAIN DISPLAY METHODS
    # --------------------------------------------------------------------------------------------------------------
    def update_main_display(self):
        # This is executed whenever we need to update the main display (change data/wavelength/etc)
        print('Updating stokes preview...')

        # clear the existing image
        # self.display_canvas_figure.clear()

        # plot stokes data
        II = self.stokes_data[0][:,:,self.displayed_wavelength]
        QQ = self.stokes_data[1][:,:,self.displayed_wavelength]
        UU = self.stokes_data[2][:,:,self.displayed_wavelength]
        VV = self.stokes_data[3][:,:,self.displayed_wavelength]

        self.display_canvas_ax[0,0].imshow(II, extent=[0,1,0,1], cmap='gray', aspect='auto')
        self.display_canvas_ax[0,1].imshow(QQ, extent=[0,1,0,1], cmap='gray', aspect='auto')
        self.display_canvas_ax[1,0].imshow(UU, extent=[0,1,0,1], cmap='gray', aspect='auto')
        self.display_canvas_ax[1,1].imshow(VV, extent=[0,1,0,1], cmap='gray', aspect='auto')
        
        # remove the ticks in the axes and set the subplots closer
        for i in range(2):
            for j in range(2):
                self.display_canvas_ax[i,j].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
                self.display_canvas_ax[i,j].set_xlim([0,1])
                self.display_canvas_ax[i,j].set_ylim([0,1])
                self.display_canvas_ax[i,j].set_xticks([])
                self.display_canvas_ax[i,j].set_yticks([])

        # refresh canvas
        self.display_canvas.draw()

    # --------------------------------------------------------------------------------------------------------------
    # DATA METHODS
    # --------------------------------------------------------------------------------------------------------------
    def update_displayed_wavelength(self, initialization=False):
        # This is executed whenever the wavelength slider is moved or the value is changed
        self.displayed_wavelength = self.wavelength_selector.value()
        self.displayed_wavelength_label.setText(f'{self.frequency_array[self.displayed_wavelength]: e} nm')
        if not initialization:
            self.plot_polygon(self.mask_array)
            self.update_main_display()

    # --------------------------------------------------------------------------------------------------------------
    # GEOMETRY METHODS
    # --------------------------------------------------------------------------------------------------------------
    def capture_geometry_coordinates(self):
        # This is executed when the button capture is pressed
        print('Capturing coordinates...')
        # initialize the list of coordinates, the click counter and the canvas and set mouse tracking to true
        self.cliks = 0
        self.geometry_coordinates = []
        self.mask_coordinates.setText('')
        self.geometry_plot_canvas.setMouseTracking(True)    # This is needed to get the mouse click events
        self.geometry_plot_canvas.mousePressEvent = self.read_geometry_coordinates
        self.plot_polygon(self.geometry_coordinates)        # plot the background image

    def save_mask_coordinates(self):
        # This is executed when the button save is pressed
        print('Saving mask coordinates...')
        if self.mask_coordinates.text() == '':
            self.geometry_coordinates = []
        print(self.geometry_coordinates)
        # set the mask coordinates to the geometry coordinates with the resolution
        mask_coordinates = [(x * int(self.x_resolution.text()), y * int(self.y_resolution.text())) for x, y in self.geometry_coordinates]
        # disable the mouse tracking
        self.geometry_plot_canvas.setMouseTracking(False)

        # if there is coordinates compute the mask
        if len(mask_coordinates) > 1:
            # Generate the mask for the polygon coordinates and the resolution
            img = Image.new('L', (int(self.x_resolution.text()), int(self.y_resolution.text())), 0)
            ImageDraw.Draw(img).polygon(mask_coordinates, outline=1, fill=1)
            # convert the image to a numpy array and save it to the input data
            self.mask_array = np.array(img)
            print(self.mask_array)
        else :
            print('No coordinates captured!')
            self.mask_array = np.zeros((int(self.x_resolution.text()), int(self.y_resolution.text())))
            print(self.mask_array)
            # plot the mask
            self.plot_polygon([])

    def read_geometry_coordinates(self, event):
        # This is executed when the mouse is clicked inside the canvas
        if self.geometry_plot_canvas.hasMouseTracking():
            # get the coordinates of the click, normalize them and add them to the list of coordinates
            self.geometry_coordinates.append((round(event.x() / self.geometry_canvas.width(), 3), round(event.y() / self.geometry_canvas.height(), 3)))
            self.cliks += 1
            print('Coordinate captured!')
            # set the coordinates in the text box
            self.mask_coordinates.setText(f'{self.geometry_coordinates}')
            # draw the polygon if the click counter is greater than 1
            if self.cliks > 1:
                self.plot_polygon(self.geometry_coordinates)

    def set_mask_coordinates(self):
        # This is executed when the coordinates are set in the text box
        print('Changing mask coordinates...')
        print(self.mask_coordinates.text())

        # get the coordinates from the text box
        try:
            self.geometry_coordinates = eval(self.mask_coordinates.text())
            if type(self.geometry_coordinates) != list:
                raise TypeError('The coordinates must be a list!')
            elif len(self.geometry_coordinates) < 2:
                raise ValueError('The list must have at least 2 coordinates to be a polygon!')
            else:
                self.plot_polygon(self.geometry_coordinates)

        except Exception as e:
            print(e)
            print('Invalid coordinates!')
            # self.plot_polygon([])

    def plot_polygon(self, coordinates):
        # Drawing the background image and the polygon (if needed)
        # clear the existing image
        self.geometry_figure.clear()

        # create an axis and set limits
        ax = self.geometry_figure.add_subplot(111)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)

        # fill the canvas with the plot
        plt.gca().set_position([0, 0, 1, 1])

        # plot the background image
        img = self.stokes_data[0,:,:,self.displayed_wavelength]
        ax.imshow(img, extent=[0,1,0,1], cmap='gray')
        plt.axis('tight')

        if type(coordinates) == list:
            print('Plotting polygon...')
            # retrieve the coordinates
            x = [p[0] for p in coordinates]
            y = [1-p[1] for p in coordinates]
            # plot data
            ax.fill(x,y, color='green', alpha=0.5)
        elif type(coordinates) == np.ndarray:
            print('Plotting mask...')
            coordinates = np.ma.masked_where(coordinates == 0, coordinates)
            ax.imshow(coordinates, extent=[0,1,0,1], cmap='PiYG', alpha=0.5,
                      vmin=-1, vmax=1, interpolation='none')
            plt.axis('tight')

        else:
            raise ValueError('The mask must be a list or a numpy array!')

        # refresh canvas
        self.geometry_plot_canvas.draw()

    # --------------------------------------------------------------------------------------------------------------
    # STOKES METHODS
    # --------------------------------------------------------------------------------------------------------------
    def update_stokes_preview(self):
        # This is executed when the number of wavelengths is changed
        print('Updating stokes preview...')

        # clear the existing image
        self.stokes_figure.clear()
        # create an axis and set limits
        ax = self.stokes_figure.add_subplot(111)

        # plot stokes data
        xx = self.frequency_array
        II = self.stokes_data[0].mean(axis=(0,1))
        QQ = self.stokes_data[1].mean(axis=(0,1))
        UU = self.stokes_data[2].mean(axis=(0,1))
        VV = self.stokes_data[3].mean(axis=(0,1))

        # plot the data
        ax.plot(xx, II, label='I', alpha=0.75)
        ax.plot(xx, QQ, label='Q', alpha=0.75)
        ax.plot(xx, UU, label='U', alpha=0.75)
        ax.plot(xx, VV, label='V', alpha=0.75)
        ax.legend()

        ax.xaxis.label.set_text('frequencies [Hz]')
        ax.yaxis.label.set_text('Intensity (normalized)')

        # refresh canvas
        self.stokes_plot_canvas.draw()

    # --------------------------------------------------------------------------------------------------------------
    # AUXILIARY METHODS
    # --------------------------------------------------------------------------------------------------------------
    def update_ui(self):
        # This is executed when the input data is changed
        print('Updating UI...')

        # get the data
        self.mask_array = np.array(self.input_data["observations"][0]["data"]["mask"])
        self.frequency_array = np.array(self.input_data["observations"][0]["data"]["frequencies"])
        self.stokes_data = np.array(self.input_data["observations"][0]["data"]["stokes_data"])

        # update the data section
        self.number_wavelengths.setText(f'{self.frequency_array.shape[0]}')
        self.x_resolution.setText(f'{self.input_data["observations"][0]["data"]["resolution"][0]}')
        self.y_resolution.setText(f'{self.input_data["observations"][0]["data"]["resolution"][1]}')
        self.fov_x.setValue(self.input_data["observations"][0]["FOV"][0])
        self.fov_y.setValue(self.input_data["observations"][0]["FOV"][1])
        self.instrument.setCurrentText(self.input_data["observations"][0]["instrument"]["name"])

        self.wavelength_selector.setValue(0)
        self.wavelength_selector.setMaximum(self.frequency_array.shape[0]-1)
        self.displayed_wavelength = 0
        self.displayed_wavelength_label.setText(f'{self.frequency_array[self.displayed_wavelength]: e} nm')
        
        # update the geometry section
        self.observer_x.setValue(self.input_data["observations"][0]["observer_r"][0])
        self.observer_y.setValue(self.input_data["observations"][0]["observer_r"][1])
        self.observer_z.setValue(self.input_data["observations"][0]["observer_r"][2])
        self.observer_phi.setValue(self.input_data["observations"][0]["orientation"][0])
        self.observer_theta.setValue(self.input_data["observations"][0]["orientation"][1])
        self.observer_psi.setValue(self.input_data["observations"][0]["orientation"][2])
        self.domain_x.setValue(self.input_data["domain"]["location"][0])
        self.domain_y.setValue(self.input_data["domain"]["location"][1])
        self.domain_z.setValue(self.input_data["domain"]["location"][2])
        self.domain_phi.setValue(self.input_data["domain"]["orientation"][0])
        self.domain_theta.setValue(self.input_data["domain"]["orientation"][1])
        self.domain_psi.setValue(self.input_data["domain"]["orientation"][2])
        self.dimension_x.setValue(self.input_data["domain"]["dimensions"][0])
        self.dimension_y.setValue(self.input_data["domain"]["dimensions"][1])
        self.dimension_z.setValue(self.input_data["domain"]["dimensions"][2])

        # update the stokes section
        self.noise_level.setValue(float(self.input_data["observations"][0]["data"]["noise"]["data"]))
        self.noise_type.setCurrentText(self.input_data["observations"][0]["data"]["noise"]["type"])
        self.I_weight.setValue(self.input_data["observations"][0]["data"]["w_IQUV"][0])
        self.Q_weight.setValue(self.input_data["observations"][0]["data"]["w_IQUV"][1])
        self.U_weight.setValue(self.input_data["observations"][0]["data"]["w_IQUV"][2])
        self.V_weight.setValue(self.input_data["observations"][0]["data"]["w_IQUV"][3])
        
        # update the atom section
        self.atomic_module.setCurrentText(self.input_data["atomic_modules"][0]["id"])

        # update the quantities section
        self.quant0.setChecked(not self.input_data["MHDLikeQ"][0]["is_constant"])

        # update the algorithm section
        self.atomic_module.setCurrentText(self.input_data["algorithm"]["id"])

        print('UI updated!')
        # update the mask preview
        self.plot_polygon(self.mask_array)
        self.update_stokes_preview()
        self.update_main_display()

    def get_gui_data(self):
        # copy the input data as default
        data = self.input_data.copy()

        # update the data section
        data["observations"][0]["data"]["frequencies"] = self.frequency_array.tolist()
        data["observations"][0]["data"]["resolution"] = [int(self.x_resolution.text()),
                                                         int(self.y_resolution.text())]
        data["observations"][0]["FOV"] = [self.fov_x.value(), self.fov_y.value()]
        data["observations"][0]["instrument"]["name"] = self.instrument.currentText()

        # update the geometry section
        data["observations"][0]["observer_r"] = [self.observer_x.value(),
                                                 self.observer_y.value(),
                                                 self.observer_z.value()]
        data["observations"][0]["orientation"] = [self.observer_phi.value(),
                                                  self.observer_theta.value(),
                                                  self.observer_psi.value()]
        data["domain"]["location"] = [self.domain_x.value(),
                                      self.domain_y.value(),
                                      self.domain_z.value()]
        data["domain"]["orientation"] = [self.domain_phi.value(),
                                         self.domain_theta.value(),
                                         self.domain_psi.value()]
        data["domain"]["dimensions"] = [self.dimension_x.value(),
                                        self.dimension_y.value(),
                                        self.dimension_z.value()]
        data["observations"][0]["data"]["mask"] = self.mask_array.tolist()

        # update the stokes section
        data["observations"][0]["data"]["stokes_data"] = self.stokes_data.tolist()
        data["observations"][0]["data"]["noise"]["data"] = self.noise_level.value()
        data["observations"][0]["data"]["noise"]["type"] = self.noise_type.currentText()
        data["observations"][0]["data"]["w_IQUV"] = [self.I_weight.value(),
                                                     self.Q_weight.value(),
                                                     self.U_weight.value(),
                                                     self.V_weight.value()]

        # update the atom section
        data["atomic_modules"][0]["id"] = self.atomic_module.currentText()

        # update the quantities section
        data["MHDLikeQ"][0]["is_constant"] = not self.quant0.isChecked()
        # data["MHDLikeQ"][1]["is_constant"] = self.quant1.isChecked()

        # update the algorithm section
        data["algorithm"]["id"] = self.algorithm.currentText()

        return data

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ui()
    sys.exit(app.exec_())
