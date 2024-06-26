import sys
from PyQt5.QtWidgets import (
    QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QDialog,
    QFormLayout, QLineEdit, QDialogButtonBox, QApplication, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsItem, QGraphicsPixmapItem,
    QMessageBox, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsTextItem,
    QGraphicsDropShadowEffect, QAction, QFileDialog, QGridLayout
)
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QFont, QDrag, QPixmap, QTransform
from PyQt5.QtCore import Qt, QMimeData, QRect, QRectF, QPointF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns
import numpy as np
from signal_generation import build_circuit_and_generate_signal, generate_ecd
from qutip import *
import csv

Nt = 1  # Number of transmons: only 1 is allowed currently

def draw_gate(gate_type, width, height, label_text, label_font_size, cavity_index=0, initial=False):
    buffer = 10  # Buffer to ensure the full image is displayed
    width = int(width)
    height = int(height)
    pixmap = QPixmap(width, height + buffer * 2)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    
    if gate_type == 'SNAP':
        painter.setBrush(QColor('#072AC8'))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, buffer, width, 50)
        painter.setPen(Qt.white)
        painter.setFont(QFont('Helvetica', label_font_size - 4, QFont.Bold))
        painter.drawText(QRect(0, buffer, width, 50), Qt.AlignCenter, label_text)
    
    elif gate_type == 'Displacement':
        painter.setBrush(QColor('#D7263D'))
        painter.setPen(Qt.NoPen)
        # Apply to one cavity
        painter.drawRect(0, buffer if initial else 60 * cavity_index + buffer, width, 50)
        painter.setPen(Qt.white)
        painter.setFont(QFont('Helvetica', label_font_size - 4, QFont.Bold))
        painter.drawText(QRect(0, buffer if initial else 60 * cavity_index + buffer, width, 50), Qt.AlignCenter, label_text)
    
    elif gate_type == 'ECD':
        painter.setBrush(QColor('#5F00BA'))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, buffer if initial else 60 * cavity_index + buffer, width, 50)
        painter.setPen(Qt.white)
        painter.setFont(QFont('Helvetica', label_font_size - 4, QFont.Bold))
        painter.drawText(QRect(0, buffer if initial else 60 * cavity_index + buffer, width, 50), Qt.AlignCenter, label_text)

    elif gate_type == 'Rotation':
        painter.setBrush(QColor('#1789FC'))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, buffer, width, 50)
        painter.setPen(Qt.white)
        painter.setFont(QFont('Helvetica', label_font_size - 4, QFont.Bold))
        painter.drawText(QRect(0, buffer, width, 50), Qt.AlignCenter, label_text)

    painter.end()
    return pixmap

class DragGate(QLabel):
    def __init__(self, gate_type, parent=None):
        super().__init__(parent)
        self.gate_type = gate_type

        if gate_type == 'SNAP':
            self.setPixmap(draw_gate('SNAP', 50, 50, 'S', 20, initial=True))
        elif gate_type == 'Displacement':
            self.setPixmap(draw_gate('Displacement', 50, 50, 'D', 20, initial=True))
        elif gate_type == 'ECD':
            self.setPixmap(draw_gate('ECD', 50, 50, 'E', 20, initial=True))
        elif gate_type == 'Rotation':
            self.setPixmap(draw_gate('Rotation', 50, 50, 'R', 20, initial=True))

        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.gate_type)
            drag.setMimeData(mime_data)

            if self.gate_type == 'SNAP':
                drag.setPixmap(draw_gate('SNAP', 50, 500, 'S', 20))
            elif self.gate_type == 'Displacement':
                drag.setPixmap(draw_gate('Displacement', 50, 500, 'D', 20))
            elif self.gate_type == 'ECD':
                drag.setPixmap(draw_gate('ECD', 50, 500, 'E', 20))
            elif self.gate_type == 'Rotation':
                drag.setPixmap(draw_gate('Rotation', 50, 500, 'R', 20))

            drag.setHotSpot(event.pos())
            drag.exec_(Qt.MoveAction)

class SystemParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Parameters")
        self.resize(450, 180)
        
        self.layout = QFormLayout(self)

        self.chi_input = QLineEdit("2*pi")
        self.chi_input.setPlaceholderText("Enter χ value (e.g., 2*pi)")
        self.chi_input.setWhatsThis("""
            <p><b>χ:</b> The dispersive shift of the system.</p>
        """)
        self.layout.addRow("χ", self.chi_input)

        self.t1_input = QLineEdit("100")
        self.t1_input.setPlaceholderText("Enter T1 value")
        self.t1_input.setWhatsThis("""
            <p><b>T1:</b> The relaxation time of the transmon in microseconds (µs).</p>
        """)
        self.layout.addRow("T1", self.t1_input)
        
        self.nc_input = QLineEdit("1")
        self.nc_input.setPlaceholderText("Enter number of cavities (Nc)")
        self.nc_input.setWhatsThis("""
            <p><b>Nc:</b> The number of cavities in the system.</p>
        """)
        self.layout.addRow("Nc", self.nc_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_parameters(self):
        chi_value = self.chi_input.text().replace("pi", str(np.pi))
        return float(eval(chi_value)), float(self.t1_input.text()), int(self.nc_input.text())

class GateParameterDialog(QDialog):
    def __init__(self, gate_type, start_time, chi, t1, Nc, parent=None, edit_mode=False):
        super().__init__(parent)
        self.setWindowTitle(f"{gate_type} Parameters")
        self.resize(760, 380)
        
        self.t1 = t1
        self.chi = chi
        self.Nc = Nc
        self.edit_mode = edit_mode
        
        self.setStyleSheet("""
            QDialog {
                font-family: 'Helvetica', sans-serif;
                font-size: 20px;
            }
            QLineEdit {
                font-family: 'Helvetica', sans-serif;
                font-size: 20px;
                padding: 10px;
            }
            QLabel {
                font-family: 'Helvetica', sans-serif;
                font-size: 20px;
            }
            QDialogButtonBox {
                font-family: 'Helvetica', sans-serif;
                font-size: 20px;
                padding: 10px;
            }
            QPushButton {
                font-family: 'Helvetica', sans-serif;
                font-size: 20px;
                padding: 10px;
            }
        """)

        self.layout = QFormLayout(self)

        self.gate_type = gate_type

        self.start_time_input = QLineEdit(self)
        self.start_time_input.setText(str(start_time))
        self.start_time_input.setWhatsThis("""
            <p><b>Start Time:</b> The time at which the pulse should begin.</p>
        """)
        if self.edit_mode:
            self.start_time_input.setDisabled(True)
        self.layout.addRow("Start Time", self.start_time_input)

        self.button_container = QWidget()
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(550, 0, 0, 0)
        self.append_button = QPushButton("Append to End")
        self.append_button.clicked.connect(self.append_to_end)
        self.button_layout.addWidget(self.append_button)
        self.layout.addRow(self.button_container)

        if gate_type == 'SNAP':
            self.thetas_input = QLineEdit(self)
            self.thetas_input.setPlaceholderText("Enter theta values separated by commas (e.g., 0, pi/2, 2*pi)")
            self.thetas_input.setWhatsThis("""
                <p><b>Thetas:</b> The SNAP gate is defined as</p>
                <p>S(θ) = Σ(exp(i θ<sub>n</sub>) |n&gt;&lt;n|)</p>
                <p>where θ<sub>n</sub> are the phase shifts applied to each Fock state |n&gt;.</p>
            """)
            self.layout.addRow("Thetas", self.thetas_input)
            self.length_factor_input = QLineEdit("10")
            self.length_factor_input.setPlaceholderText("Enter SNAP pulse length factor")
            self.length_factor_input.setWhatsThis("""
                <p><b>Length Factor:</b> Determines the duration of the pulse.</p>
                <p>Pulse length = (2π / χ) · length factor.</p>
            """)            
            if self.Nc > 1:
                self.cavity_indices_input = QLineEdit(self)
                self.cavity_indices_input.setPlaceholderText(f"Enter cavity indices separated by commas")
                self.cavity_indices_input.setWhatsThis("""
                    <p><b>Cavity Indices:</b> Specifies which cavity or cavities the SNAP gate should target.</p>
                """)
                self.layout.addRow("Cavity Indices", self.cavity_indices_input)
                # if self.edit_mode:
                #     self.cavity_indices_input.setDisabled(True)
            self.layout.addRow("Length Factor", self.length_factor_input)
        
        elif gate_type == 'Displacement':
            self.alpha_input = QLineEdit(self)
            self.alpha_input.setWhatsThis("""
                <p><b>Alpha:</b> The Displacement operator is defined as</p>
                <p>D(α) = exp(α a<sup>†</sup> - α<sup>*</sup> a)</p>
                <p>where α is the displacement amplitude.</p>
            """)
            self.layout.addRow("Alpha", self.alpha_input)
            self.length_factor_input = QLineEdit("0.01")
            self.length_factor_input.setPlaceholderText("Enter Displacement pulse length factor")
            self.length_factor_input.setWhatsThis("""
                <p><b>Length Factor:</b> Determines the duration of the pulse.</p>
                <p>Pulse length = (2π / χ) · length factor.</p>
            """)  
            if self.Nc > 1:
                self.cavity_index_input = QLineEdit(self)
                self.cavity_index_input.setPlaceholderText("Enter cavity index")
                self.cavity_index_input.setWhatsThis("""
                    <p><b>Cavity Index:</b> Specifies which cavity the Displacement operator should target.</p>
                """)
                self.layout.addRow("Cavity Index", self.cavity_index_input)
                # if self.edit_mode:
                #     self.cavity_index_input.setDisabled(True)
            self.layout.addRow("Length Factor", self.length_factor_input)
            
        elif gate_type == 'ECD':
            self.beta_input = QLineEdit(self)
            self.beta_input.setWhatsThis("""
                <p><b>Beta:</b> The ECD gate is defined as</p>
                <p>ECD(β) = D(β / 2) |e&gt;&lt;g| + D(-β / 2) |g&gt;&lt;e|</p>
                <p>where β is the displacement amplitude.</p>
            """)
            self.layout.addRow("Beta", self.beta_input)
            self.unit_amp_input = QLineEdit("0.05")
            self.unit_amp_input.setPlaceholderText("Enter unit amplitude")
            self.unit_amp_input.setWhatsThis("""
                <p><b>Unit Amplitude:</b> The unit amplitude scaling of the ECD pulse.</p>
            """)
            self.layout.addRow("Unit Amplitude", self.unit_amp_input)
            self.max_amp_input = QLineEdit("30")
            self.max_amp_input.setPlaceholderText("Enter maximum amplitude (α<sub>0<sub>)")
            self.max_amp_input.setWhatsThis("""
                <p><b>Max Amplitude:</b> The maximum displacement amplitude for the ECD pulse (α<sub>0<sub>).</p>
            """)
            self.layout.addRow("Max Amplitude", self.max_amp_input)
            if self.Nc > 1:
                self.cavity_index_input = QLineEdit(self)
                self.cavity_index_input.setPlaceholderText("Enter cavity index")
                self.cavity_index_input.setWhatsThis("""
                    <p><b>Cavity Index:</b> Specifies which cavity the ECD gate should target.</p>
                """)
                self.layout.addRow("Cavity Index", self.cavity_index_input)
                # if self.edit_mode:
                #     self.cavity_index_input.setDisabled(True)

        elif gate_type == 'Rotation':
            self.theta_input = QLineEdit(self)
            self.theta_input.setWhatsThis("""
                <p><b>Theta:</b> The Rotation operator is defined as</p>
                <p>R<sub>φ</sub>(θ) = exp(-i(θ / 2)(σ<sub>x</sub> cos φ + σ<sub>y</sub> sin φ))</p>
                <p>where θ is the rotation angle.</p>
            """)
            self.layout.addRow("Theta", self.theta_input)

            self.phi_input = QLineEdit(self)
            self.phi_input.setWhatsThis("""
                <p><b>Phi:</b> The rotation operator is defined as</p>
                <p>R<sub>φ</sub>(θ) = exp(-i(θ / 2)(σ<sub>x</sub> cos φ + σ<sub>y</sub> sin φ))</p>
                <p>where φ is the phase angle.</p>
            """)
            self.layout.addRow("Phi", self.phi_input)

            self.length_factor_input = QLineEdit("0.01")
            self.length_factor_input.setPlaceholderText("Enter Rotation pulse length factor")
            self.length_factor_input.setWhatsThis("""
                <p><b>Length Factor:</b> Determines the duration of the pulse.</p>
                <p>Pulse length = (2π / χ) · length factor.</p>
            """)
            self.layout.addRow("Length Factor", self.length_factor_input)
            
            self.unit_amp_input = QLineEdit("0.05")
            self.unit_amp_input.setPlaceholderText("Enter unit amplitude")
            self.unit_amp_input.setWhatsThis("""
                <p><b>Unit Amplitude:</b> The unit amplitude scaling of the Rotation pulse.</p>
            """)
            self.layout.addRow("Unit Amplitude", self.unit_amp_input)

        if gate_type != 'Rotation':
            self.phase_input = QLineEdit("0")
            self.phase_input.setPlaceholderText("Enter phase value (e.g., 0, pi/2, 2*pi)")
            self.phase_input.setWhatsThis("""
                <p><b>Phase:</b> The additional phase to apply to the pulse.</p>
            """)
            self.layout.addRow("Phase", self.phase_input)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_parameters(self):
        parameters = {
            "t_i": self.validate_float(self.start_time_input.text(), "start time"),
        }
        if self.gate_type != 'Rotation': parameters["phase"] = self.validate_float(self.phase_input.text(), "phase")

        two_pi = 2 * np.pi

        if self.gate_type == 'SNAP':
            thetas = self.validate_float_list(self.thetas_input.text(), "theta values")
            parameters["thetas"] = thetas
            if self.Nc == 1:
                parameters["cavity_indices"] = [1]
            else:
                cavity_indices = self.validate_int_list(self.cavity_indices_input.text(), "cavity indices")
                if any(index < 1 or index > self.Nc for index in cavity_indices):
                    self.showError(f"Invalid cavity indices. Please enter integers between 1 and {self.Nc}.")
                    raise ValueError(f"Invalid cavity indices. Please enter integers between 1 and {self.Nc}.")
                parameters["cavity_indices"] = list(set(cavity_indices))  # Remove duplicates
            length_factor = self.validate_float(self.length_factor_input.text(), "length factor")
            parameters["delta_t"] = (two_pi / self.chi) * length_factor
            
        elif self.gate_type == 'Displacement':
            alpha = self.alpha_input.text().split(',')
            if len(alpha) > 1:
                self.showError(f"Only one alpha value is allowed for Displacement operator.")
                raise ValueError(f"Only one alpha value is allowed for Displacement operator.")
            parameters["alpha"] = self.validate_complex(alpha[0], "alpha")
            length_factor = self.validate_float(self.length_factor_input.text(), "length factor")
            parameters["delta_t"] = (two_pi / self.chi) * length_factor  # Duration
            if self.Nc == 1:
                parameters["cavity_index"] = 1
            else:
                cavity_index = self.validate_int(self.cavity_index_input.text(), "cavity index")
                if cavity_index < 1 or cavity_index > self.Nc:
                    self.showError(f"Invalid cavity index. Please enter an integer between 1 and {self.Nc}.")
                    raise ValueError(f"Invalid cavity index. Please enter an integer between 1 and {self.Nc}.")
                parameters["cavity_index"] = cavity_index

        elif self.gate_type == 'ECD':
            beta = self.beta_input.text().split(',')
            if len(beta) > 1:
                self.showError(f"Only one beta value is allowed for ECD gate.")
                raise ValueError(f"Only one beta value is allowed for ECD gate.")
            parameters["beta"] = self.validate_complex(beta[0], "beta")
            parameters["unit_amp"] = self.validate_float(self.unit_amp_input.text(), "unit_amp")
            parameters["max_amp"] = self.validate_float(self.max_amp_input.text(), "max_amp")
            if self.Nc == 1:
                parameters["cavity_index"] = 1
            else:
                cavity_index = self.validate_int(self.cavity_index_input.text(), "cavity index")
                if cavity_index < 1 or cavity_index > self.Nc:
                    self.showError(f"Invalid cavity index. Please enter an integer between 1 and {self.Nc}.")
                    raise ValueError(f"Invalid cavity index. Please enter an integer between 1 and {self.Nc}.")
                parameters["cavity_index"] = cavity_index

        elif self.gate_type == 'Rotation':
            theta = self.theta_input.text().split(',')
            if len(theta) > 1:
                self.showError(f"Only one theta value is allowed for Rotation operator.")
                raise ValueError(f"Only one theta value is allowed for Rotation operator.")
            parameters["theta"] = self.validate_float(self.theta_input.text(), "theta")
            phi = self.phi_input.text().split(',')
            if len(phi) > 1:
                self.showError(f"Only one phi value is allowed for Rotation operator.")
                raise ValueError(f"Only one phi value is allowed for Rotation operator.")
            parameters["phi"] = self.validate_float(self.phi_input.text(), "phi")
            length_factor = self.validate_float(self.length_factor_input.text(), "length factor")
            parameters["delta_t"] = (two_pi / self.chi) * length_factor  # Duration
            parameters["unit_amp"] = self.validate_float(self.unit_amp_input.text(), "unit_amp")
        
        return parameters
    
    def append_to_end(self):
        main_window = self.get_main_window()
        latest_end_time = 0
        for item in main_window.scene.items():
            if isinstance(item, CircuitWire):
                for gate, params, _, _, _, _ in item.gates:
                    end_time = params['t_i'] + params['delta_t']
                    if end_time > latest_end_time:
                        latest_end_time = end_time
        
        self.start_time_input.setText(str(latest_end_time))
        self.start_time_input.setFocus()

    def get_main_window(self):
        view = self.parent()
        while not isinstance(view, QMainWindow) and view is not None:
            view = view.parent()
        return view

    def validate_float(self, value, name):
        try:
            value = value.replace("pi", str(np.pi))
            return float(eval(value))
        except (ValueError, SyntaxError):
            self.showError(f"Invalid {name}. Please enter a valid number or an expression involving 'pi'.")
            raise ValueError(f"Invalid {name}. Please enter a valid number or an expression involving 'pi'.")
    
    def validate_int(self, value, name):
        try:
            return int(value)
        except ValueError:
            self.showError(f"Invalid {name}. Please enter a valid integer.")
            raise ValueError(f"Invalid {name}. Please enter a valid integer.")
    
    def validate_float_list(self, value, name):
        if not value.strip():
            self.showError(f"Invalid {name}. Please enter a list of valid numbers separated by commas.")
            raise ValueError(f"Invalid {name}. Please enter a list of valid numbers separated by commas.")
        try:
            return [float(eval(item.replace("pi", str(np.pi)))) for item in value.split(",")]
        except ValueError:
            self.showError(f"Invalid {name}. Please enter a list of valid numbers separated by commas.")
            raise ValueError(f"Invalid {name}. Please enter a list of valid numbers separated by commas.")
    
    def validate_int_list(self, value, name):
        if not value.strip():
            self.showError(f"Invalid {name}. Please enter a list of valid integers separated by commas.")
            raise ValueError(f"Invalid {name}. Please enter a list of valid integers separated by commas.")
        try:
            return [int(item) for item in value.split(",")]
        except ValueError:
            self.showError(f"Invalid {name}. Please enter a list of valid integers separated by commas.")
            raise ValueError(f"Invalid {name}. Please enter a list of valid integers separated by commas.")

    def validate_complex(self, value, name):
        try:
            value = value.replace("pi", str(np.pi))
            complex_val = complex(eval(value.replace('i', 'j')))
            return complex_val
        except (ValueError, SyntaxError):
            self.showError(f"Invalid {name}. Please enter a valid complex number (e.g., pi, 1+2i, or 3-4j).")
            raise ValueError(f"Invalid {name}. Please enter a valid complex number (e.g., pi, 1+2i, or 3-4j).")
    
    def showError(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

class CircuitWire(QGraphicsItem):
    def __init__(self, wire_type, index, t1, Nc):
        super().__init__()
        self.wire_type = wire_type
        self.index = index
        self.t1 = t1
        self.Nc = Nc
        self.gates = []
        self.initUI()
        self.setAcceptDrops(True)
        self.offset = 150  # Offset to align start of time and wires
        self.setAcceptHoverEvents(True)
        self.gate_positions = []
        
    def initUI(self):
        self.gate_positions = []
        
    def boundingRect(self):
        return QRectF(0, 0, int(self.t1 * 50) + self.offset, (self.Nc + 1) * 60)

    def paint(self, painter, option, widget):
        pen = QPen(QColor(42, 43, 42, int(255 * 0.95)), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(40, 25, int(self.t1 * 50) + 50, 25)
        painter.setFont(QFont('Helvetica', 14, QFont.Bold))
        painter.drawText(QRectF(-50, 0, 40, 50), Qt.AlignCenter, self.wire_type[0] + (str(self.index) if self.index > 0 else ''))

    def get_main_window(self):
        view = self.scene().views()[0]
        while view.parent() is not None:
            view = view.parent()
        return view

    def addGate(self, gate, position):
        visualization_start_time = (position.x() - self.offset) / 50  # 50 pixels is one microsecond
        start_time = 0.0 if visualization_start_time < 0.0 else position.x() / 50
        
        parent_widget = self.scene().views()[0]
        main_window = self.get_main_window()
        dialog = GateParameterDialog(gate, start_time, main_window.chi, main_window.t1, main_window.Nc, parent_widget)
        if dialog.exec_() == QDialog.Accepted:
            try:
                parameters = dialog.get_parameters()
            except ValueError as e:
                print(e)
                return
            
            start_time = parameters['t_i']  
            visualization_position = start_time * 50 + self.offset

            if gate == 'ECD':
                _, _, _, _, length = generate_ecd(
                    parameters['beta'], main_window.chi, start_time, 
                    parameters['phase'], parameters['unit_amp'], parameters['max_amp']
                )
                parameters['delta_t'] = length
            
            duration = parameters['delta_t']
            end_time = start_time + duration

            if duration <= 0:
                self.showError("Duration must be greater than zero.")
                return

            # Check for overlaps with existing gates on all wires
            for item in self.scene().items():
                if isinstance(item, CircuitWire):
                    for existing_gate, existing_params, _, _, _, _ in item.gates:
                        existing_start_time = existing_params['t_i']
                        existing_duration = existing_params['delta_t']
                        existing_end_time = existing_start_time + existing_duration
                        if gate == 'Displacement' and existing_gate == 'Displacement':
                            continue  # Allow multiple displacements to different cavities
                        if not (end_time <= existing_start_time or start_time >= existing_end_time):
                            self.showError("Gate overlap detected. Choose a different time.")
                            return

            if end_time > self.t1:
                self.showError(f"Total gate sequence duration exceeds T1 ({self.t1} µs). Choose a different time.")
                return

            gate_width = max(int(duration * 50), 30)  # Visualization - 50 pixels per microsecond, minimum 30 pixels

            additional_graphics = []
            if gate == 'SNAP':
                gate_item = QGraphicsPixmapItem(draw_gate('SNAP', gate_width, 500, 'S', 20))
                # Draw wire to the selected cavities
                cavity_indices = parameters["cavity_indices"]
                max_cavity_index = max(cavity_indices)
                pen = QPen(QColor(48, 54, 51, int(255 * 0.95)), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                line = self.scene().addLine(visualization_position + gate_width // 2, 50, visualization_position + gate_width // 2, max_cavity_index * 60 + 25, pen)
                additional_graphics.append(line)
                for cavity_index in cavity_indices:
                    dot = QGraphicsEllipseItem(visualization_position + gate_width // 2 - 3, cavity_index * 60 + 22, 6, 6)
                    dot.setBrush(QBrush(QColor(48, 54, 51, int(255 * 0.95))))
                    dot.setPen(QPen(Qt.NoPen))
                    self.scene().addItem(dot)
                    additional_graphics.append(dot)
            elif gate == 'Displacement':
                cavity_index = parameters["cavity_index"]
                gate_item = QGraphicsPixmapItem(draw_gate('Displacement', gate_width, 500, 'D', 20, cavity_index=cavity_index))
                gate_item.setPos(visualization_position, cavity_index * 60 - 10)
            elif gate == 'ECD':
                cavity_index = parameters["cavity_index"]
                gate_item = QGraphicsPixmapItem(draw_gate('ECD', gate_width, 500, 'E', 20, cavity_index=cavity_index))
                gate_item.setPos(visualization_position, cavity_index * 60 - 10)
                # Draw the vertical wire from the transmon wire to the top of the ECD block
                pen = QPen(QColor(48, 54, 51, int(255 * 0.95)), 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                line = self.scene().addLine(visualization_position + gate_width // 2, 25, visualization_position + gate_width // 2, cavity_index * 60, pen)
                additional_graphics.append(line)
                dot = QGraphicsEllipseItem(visualization_position + gate_width // 2 - 3, 25 - 3, 6, 6)
                dot.setBrush(QBrush(QColor(48, 54, 51, int(255 * 0.95))))
                dot.setPen(QPen(Qt.NoPen))
                self.scene().addItem(dot)
                additional_graphics.append(dot)
            elif gate == 'Rotation':
                gate_item = QGraphicsPixmapItem(draw_gate('Rotation', gate_width, 500, 'S', 20))

            gate_item.setPos(visualization_position, -10)
            self.scene().addItem(gate_item)
            gate_scene_pos = gate_item.mapToScene(gate_item.boundingRect().topLeft())
            self.gates.append((gate, parameters, position, additional_graphics, gate_item, gate_scene_pos))
            self.gate_positions.append(gate_item)
            self.get_main_window().undo_stack.append(("add", (gate, parameters, position, additional_graphics, gate_item, gate_scene_pos)))
            self.update()
            self.get_main_window().generate_signal()

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.setDropAction(Qt.MoveAction)
            event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasText():
            gate_type = event.mimeData().text()
            position = event.pos()
            column_x = round(position.x() / 50) * 50  # Align the block's position horizontally
            nearest_wire_index = self.find_nearest_wire(position.y())
            self.addGate(gate_type, QPointF(column_x, nearest_wire_index * 200))
            event.setDropAction(Qt.MoveAction)
            event.accept()

    def hoverEnterEvent(self, event):
        hovered_item = self.scene().itemAt(event.scenePos(), QTransform())
        for gate, _, _, _, gate_item, gate_scene_pos in self.gates:
            gate_scene_rect = QRectF(gate_scene_pos, gate_item.boundingRect().size())
            if gate_scene_rect.contains(event.scenePos()):
                gate_item.setOpacity(0.9)
                event.accept()
                return

    def hoverLeaveEvent(self, event):
        for _, _, _, _, gate_item, _ in self.gates:
            if gate_item != self.get_main_window().selected_gate:
                gate_item.setOpacity(1)
        event.accept()

    def mousePressEvent(self, event):
        clicked_item = self.scene().itemAt(event.scenePos(), QTransform())
        for gate, _, _, _, gate_item, gate_scene_pos in self.gates:
            gate_scene_rect = QRectF(gate_scene_pos, gate_item.boundingRect().size())
            if gate_scene_rect.contains(event.scenePos()):
                main_window = self.get_main_window()
                if main_window.selected_gate and main_window.selected_gate != gate_item:
                    main_window.selected_gate.setGraphicsEffect(None)
                    main_window.selected_gate.setOpacity(1)
                main_window.selected_gate = gate_item
                effect = QGraphicsDropShadowEffect()
                effect.setColor(QColor('#959793'))
                effect.setBlurRadius(2)
                effect.setOffset(2, 2)
                gate_item.setGraphicsEffect(effect)
                gate_item.setOpacity(0.9)
                event.accept()
                return
        # Clicking outside any gates
        main_window = self.get_main_window()
        if main_window.selected_gate:
            main_window.selected_gate.setGraphicsEffect(None)
            main_window.selected_gate.setOpacity(1)
            main_window.selected_gate = None
        super().mousePressEvent(event)

    def restoreGate(self, gate_tuple):
        gate, parameters, position, additional_graphics, gate_item, gate_scene_pos = gate_tuple
        self.gates.append(gate_tuple)
        self.gate_positions.append(gate_item)
        self.scene().addItem(gate_item)
        for graphic in additional_graphics:
            self.scene().addItem(graphic)
        self.update()

    def deleteGate(self, gate_item):
        for gate_tuple in self.gates:
            if gate_tuple[4] == gate_item:
                self.scene().removeItem(gate_item)
                for graphic in gate_tuple[3]:
                    self.scene().removeItem(graphic)
                self.gates.remove(gate_tuple)
                self.update()
                return gate_tuple
        return None
    
    def clear_all_gates(self):
        for gate, parameters, position, additional_graphics, gate_item, gate_scene_pos in self.gates:
            self.scene().removeItem(gate_item)
            for graphic in additional_graphics:
                self.scene().removeItem(graphic)
        self.gates.clear()
        self.update()

    def find_nearest_wire(self, y_position):
        wire_positions = [wire.pos().y() for wire in self.scene().items() if isinstance(wire, CircuitWire)]
        nearest_wire_y = min(wire_positions, key=lambda y: abs(y - y_position))
        nearest_wire_index = wire_positions.index(nearest_wire_y)
        return nearest_wire_index
    
    def showError(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

class SignalPlotCanvas(FigureCanvas):
    sns.set_theme(style='whitegrid', palette=sns.color_palette(), font='DejaVu Sans', font_scale=1.25)

    def __init__(self, parent=None):
        fig, (self.ax1, self.ax2) = plt.subplots(2, 1, sharex=True, figsize=(30, 8))
        fig.subplots_adjust(left=0.1, right=0.95, top=0.925, bottom=0.1)
        super(SignalPlotCanvas, self).__init__(fig)
        self.setParent(parent)
        self.draw_empty_plot()

    def draw_empty_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot([], [])
        self.ax2.plot([], [])
        self.ax1.set_title("Transmon Signal Visualization")
        self.ax1.set_ylabel("Ω(t)/2π [MHz]")
        self.ax2.set_title("Cavity Signal Visualization")
        self.ax2.set_xlabel("Time [µs]")
        self.ax2.set_ylabel("ε(t)/2π [MHz]")
        self.draw()

    def update_plot(self, tlist, transmon_I, transmon_Q, cavity_signals):
        self.ax1.clear()
        self.ax2.clear()

        # Adjust the time axis to be larger than the end time of the sequence
        self.ax1.set_xlim(0, max(tlist) * 1.1)
        self.ax2.set_xlim(0, max(tlist) * 1.1)

        # Normalize signals by 2π
        transmon_I = np.array(transmon_I)
        transmon_Q = np.array(transmon_Q)
        cavity_signals = [(np.array(cavity_I), np.array(cavity_Q)) for cavity_I, cavity_Q in cavity_signals]
        transmon_I_normalized = transmon_I / (2 * np.pi)
        transmon_Q_normalized = transmon_Q / (2 * np.pi)
        cavity_signals_normalized = [(cavity_I / (2 * np.pi), cavity_Q / (2 * np.pi)) for cavity_I, cavity_Q in cavity_signals]

        sns.lineplot(x=tlist, y=transmon_I_normalized, ax=self.ax1, label='Transmon I')
        sns.lineplot(x=tlist, y=transmon_Q_normalized, ax=self.ax1, label='Transmon Q')
        self.ax1.legend()
        self.ax1.set_title("Transmon Signal Visualization")
        self.ax1.set_ylabel("Ω(t)/2π [MHz]")

        for i, (cavity_I, cavity_Q) in enumerate(cavity_signals_normalized):
            sns.lineplot(x=tlist, y=cavity_I, ax=self.ax2, label=f'Cavity {i+1} I')
            sns.lineplot(x=tlist, y=cavity_Q, ax=self.ax2, label=f'Cavity {i+1} Q')
        self.ax2.legend()
        self.ax2.set_title("Cavity Signal Visualization")
        self.ax2.set_xlabel("Time [µs]")
        self.ax2.set_ylabel("ε(t)/2π [MHz]")

        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.selected_gate = None
        self.undo_stack = []
        self.redo_stack = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Quantum Control Generator')
        self.setGeometry(60, 80, 1600, 1000)  # Window size

        self.chi = None
        self.t1 = None
        self.Nc = None
        self.chi_expr = None  # Store chi expression
        self.t1_expr = None   # Store T1 expression

        self.get_initial_parameters()
        
        self.main_layout = QVBoxLayout()
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)

        self.snap_gate = DragGate('SNAP')
        self.displacement_operator = DragGate('Displacement')
        self.ecd_gate = DragGate('ECD')
        self.rotation_operator = DragGate('Rotation')

        gate_layout = QGridLayout()
        gate_layout.addWidget(self.snap_gate, 0, 0)
        snap_label = QLabel("SNAP")
        snap_label.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 20px;")
        gate_layout.addWidget(snap_label, 0, 1)

        gate_layout.addWidget(self.displacement_operator, 1, 0)
        displacement_label = QLabel("Displacement")
        displacement_label.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 20px;")
        gate_layout.addWidget(displacement_label, 1, 1)

        gate_layout.addWidget(self.ecd_gate, 2, 0)
        ecd_label = QLabel("ECD")
        ecd_label.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 20px;")
        gate_layout.addWidget(ecd_label, 2, 1)

        gate_layout.addWidget(self.rotation_operator, 3, 0)
        rotation_label = QLabel("Rotation")
        rotation_label.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 20px;")
        gate_layout.addWidget(rotation_label, 3, 1)

        gate_layout.setVerticalSpacing(20)

        self.parameter_widget = self.create_parameter_widget()

        vertical_layout = QVBoxLayout()
        vertical_layout.addLayout(gate_layout)
        vertical_layout.addWidget(self.parameter_widget)

        self.gate_widget = QWidget()
        self.gate_widget.setLayout(vertical_layout)
        self.gate_widget.setStyleSheet("background: none;") 
        self.gate_widget.setMinimumWidth(250)
        self.gate_widget.layout().setContentsMargins(0, 0, 30, 0)
        
        self.save_button = QPushButton('Save Signals')
        self.save_button.setStyleSheet("""
            background-color: #4DAA57;
            color: white;
            border-radius: 10px;
            padding: 10px;
            min-width: 150px;
            font-weight: bold;
            font-family: 'Helvetica', sans-serif;
            font-size: 20px;
        """)
        self.save_button.clicked.connect(self.save_signal)

        self.toolbar = self.addToolBar('Tools')
        
        undo_action = QAction('Undo', self)
        undo_action.triggered.connect(self.undo)
        self.toolbar.addAction(undo_action)

        redo_action = QAction('Redo', self)
        redo_action.triggered.connect(self.redo)
        self.toolbar.addAction(redo_action)

        delete_action = QAction('Delete Selected Gate', self)
        delete_action.triggered.connect(self.delete_selected_gate)
        self.toolbar.addAction(delete_action)

        clear_all_action = QAction('Clear All', self)
        clear_all_action.triggered.connect(self.clear_all)
        self.toolbar.addAction(clear_all_action)

        edit_action = QAction('Edit Selected Gate', self)
        edit_action.triggered.connect(self.edit_selected_gate)
        self.toolbar.addAction(edit_action)

        self.circuit_area = QGraphicsView()
        self.scene = QGraphicsScene()
        self.circuit_area.setScene(self.scene)
        self.circuit_area.setStyleSheet("""
            QGraphicsView {
                background-color: #eeeeee;
                border: 2px solid #cccccc;
                padding: 10px;
                border-radius: 10px;
            }
            QScrollBar:horizontal {
                border: 1px solid #cccccc;
                background: #ffffff;
                height: 15px;
                margin: 0px 20px 0 20px;
            }
            QScrollBar::handle:horizontal {
                background: #d6edff;
                min-width: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                background: none;
                border: none;
            }
        """)
        self.circuit_area.setInteractive(True)
        self.circuit_area.setMinimumSize(1200, 600)
        self.circuit_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.circuit_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.signal_visualization = SignalPlotCanvas(self)
        self.signal_visualization.setMinimumSize(1200, 720)
        
        main_horizontal_layout = QHBoxLayout()
        main_horizontal_layout.addWidget(self.gate_widget)
        main_horizontal_layout.addWidget(self.circuit_area)

        self.main_layout.addLayout(main_horizontal_layout)
        self.main_layout.addWidget(self.signal_visualization)
        self.main_layout.addWidget(self.save_button)

        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

        self.init_circuit_wires()
        
        # Set the total circuit length to T1 (in microseconds)
        scene_width = self.t1 * 50  # 50 pixels per microsecond
        self.scene.setSceneRect(0, 0, scene_width + 200, self.Nc * 70)

    def get_initial_parameters(self):
        dialog = SystemParameterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.chi, self.t1, self.Nc = dialog.get_parameters()
            self.chi_expr = dialog.chi_input.text()  # Save chi expression
            self.t1_expr = dialog.t1_input.text()    # Save T1 expression
        else:
            QApplication.quit()
            sys.exit()
            
    def create_parameter_widget(self):
        widget = QWidget()
        layout = QVBoxLayout()

        if 'pi' in self.chi_expr:
            self.chi_expr = self.chi_expr.replace('*pi', 'π')
            self.chi_expr = self.chi_expr.replace('* pi', 'π')
        self.chi_label = QLabel(f"χ: {self.chi_expr}")
        self.chi_label.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 20px;")
        layout.addWidget(self.chi_label)
        
        self.t1_label = QLabel(f"T1: {self.t1_expr} µs")
        self.t1_label.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 20px;")
        layout.addWidget(self.t1_label)

        # explanation = QLabel("Length factors convert to durations as follows:\n"
        #                     "Pulse length = (2π / χ) · pulse length factor")
        # explanation.setStyleSheet("font-family: 'Helvetica', sans-serif; font-size: 16px;")
        # layout.addWidget(explanation)

        widget.setLayout(layout)
        widget.setStyleSheet("background: none;")
        return widget

    def init_circuit_wires(self):
        self.transmon_wire = CircuitWire("Transmon", 0, self.t1, self.Nc)
        self.transmon_wire.setPos(100, 0)
        self.scene.addItem(self.transmon_wire)

        for i in range(self.Nc):
            cavity_wire = CircuitWire(f"Cavity {i + 1}", i + 1, self.t1, self.Nc)
            cavity_wire.setPos(100, (i + 1) * 60)
            self.scene.addItem(cavity_wire)

        self.draw_time_axis()

    def draw_time_axis(self):
        offset = 100
        axis_start = 50 + offset
        axis_end = 50 + self.t1 * 50 + offset
        axis = QGraphicsLineItem(axis_start, (self.Nc + 1) * 60 + 20, axis_end, (self.Nc + 1) * 60 + 20)
        self.scene.addItem(axis)

        tick_interval = 500  # Pixels per tick, each tick is 10 units of time
        for x in range(int(axis_start), int(axis_end) + 1, tick_interval):
            tick = QGraphicsLineItem(x, (self.Nc + 1) * 60 + 15, x, (self.Nc + 1) * 60 + 25)
            self.scene.addItem(tick)
            time_value = (x - 50 - offset) / 50
            label = QGraphicsTextItem(f"{time_value:.1f}")
            label.setPos(x - 15, (self.Nc + 1) * 60 + 30)
            self.scene.addItem(label)

    def save_signal(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getSaveFileName(self, "Save Signal", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if not filename:
            return

        pulse_params = []

        for wire in self.scene.items():
            if isinstance(wire, CircuitWire):
                for gate, params, _, _, _, _ in wire.gates:
                    if gate == 'SNAP':
                        pulse_params.append({
                            "type": "snap",
                            "parameters": params['thetas'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "phase": params['phase'],
                            "waveform": "gaussian",
                            "chop": 6,
                            "cavity_indices": [index - 1 for index in params['cavity_indices']]
                        })
                    elif gate == 'Displacement':
                        pulse_params.append({
                            "type": "displacement",
                            "parameter": params['alpha'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "phase": params['phase'],
                            "waveform": "gaussian",
                            "chop": 6,
                            "cavity_index": params["cavity_index"]
                        })
                    elif gate == 'ECD':
                        pulse_params.append({
                            "type": "ecd",
                            "parameter": params['beta'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "unit_amp": params['unit_amp'],
                            "max_amp": params['max_amp'],
                            "phase": params['phase'],
                            "waveform": "gaussian",
                            "chop": 6,
                            "cavity_index": params["cavity_index"]
                        })
                    elif gate == 'Rotation':
                        pulse_params.append({
                            "type": "rotation",
                            "theta": params['theta'],
                            "phi": params['phi'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "unit_amp": params['unit_amp'],
                            "waveform": "gaussian",
                            "chop": 6
                        })

        if not pulse_params:
            print("No valid pulse parameters found.")
            return

        control_signals = build_circuit_and_generate_signal(self.Nc, self.chi, pulse_params, plot_signals=False)
        
        end_time = max([pulse_param["t_f"] for pulse_param in pulse_params])
        tlist_end = end_time + 0.5
        tlist = np.linspace(0, tlist_end, int(tlist_end * 1000))  # Pad with zero at the end for 0.5 microseconds; resolution: 1 nanosecond
        transmon_I = [control_signals['transmon_I'](t) for t in tlist]
        transmon_Q = [control_signals['transmon_Q'](t) for t in tlist]
        cavity_signals = []
        for i in range(self.Nc):
            cavity_I = [control_signals[f'cavity_{i+1}_I'](t) for t in tlist]
            cavity_Q = [control_signals[f'cavity_{i+1}_Q'](t) for t in tlist]
            cavity_signals.append((cavity_I, cavity_Q))

        export_signal_to_csv(filename, tlist, transmon_I, transmon_Q, cavity_signals)

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(f"Signal saved to {filename}")
        msg_box.setWindowTitle("Save Confirmation")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def generate_signal(self):
        pulse_params = []

        for wire in self.scene.items():
            if isinstance(wire, CircuitWire):
                for gate, params, _, _, _, _ in wire.gates:
                    if gate == 'SNAP':
                        pulse_params.append({
                            "type": "snap",
                            "parameters": params['thetas'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "phase": params['phase'],
                            "waveform": "gaussian",
                            "chop": 6,
                            "cavity_indices": [index for index in params['cavity_indices']]
                        })
                    elif gate == 'Displacement':
                        pulse_params.append({
                            "type": "displacement",
                            "parameter": params['alpha'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "phase": params['phase'],
                            "waveform": "gaussian",
                            "chop": 6,
                            "cavity_index": params["cavity_index"]
                        })
                    elif gate == 'ECD':
                        pulse_params.append({
                            "type": "ecd",
                            "parameter": params['beta'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "unit_amp": params['unit_amp'],
                            "max_amp": params['max_amp'],                            
                            "phase": params['phase'],
                            "waveform": "gaussian",
                            "chop": 6,
                            "cavity_index": params["cavity_index"]
                        })
                    elif gate == 'Rotation':
                        pulse_params.append({
                            "type": "rotation",
                            "theta": params['theta'],
                            "phi": params['phi'],
                            "t_i": params['t_i'],
                            "t_f": params['t_i'] + params['delta_t'],
                            "unit_amp": params['unit_amp'],
                            "waveform": "gaussian",
                            "chop": 6
                        })

        if pulse_params:
            control_signals = build_circuit_and_generate_signal(self.Nc, self.chi, pulse_params, plot_signals=False)
            
            end_time = max([pulse_param["t_f"] for pulse_param in pulse_params])
            tlist_end = end_time + 0.5
            tlist = np.linspace(0, tlist_end, int(tlist_end * 1000))  # Pad with zero at the end for 0.5 microseconds; resolution: 1 nanosecond

            transmon_I = [control_signals['transmon_I'](t) for t in tlist]
            transmon_Q = [control_signals['transmon_Q'](t) for t in tlist]

            cavity_signals = []
            for i in range(self.Nc):
                cavity_I = [control_signals[f'cavity_{i+1}_I'](t) for t in tlist]
                cavity_Q = [control_signals[f'cavity_{i+1}_Q'](t) for t in tlist]
                cavity_signals.append((cavity_I, cavity_Q))

            self.signal_visualization.update_plot(tlist, transmon_I, transmon_Q, cavity_signals)
        else:
            self.signal_visualization.draw_empty_plot()  # No gates, clear the plot
    
    def undo(self):
        if self.undo_stack:
            last_action = self.undo_stack.pop()
            action_type, gate_tuple = last_action
            gate, parameters, position, additional_graphics, gate_item, gate_scene_pos = gate_tuple
            if action_type == "add":
                for wire in self.scene.items():
                    if isinstance(wire, CircuitWire):
                        wire.deleteGate(gate_item)
            elif action_type == "delete":
                for wire in self.scene.items():
                    if isinstance(wire, CircuitWire) and wire.index == (parameters.get('cavity_index') or parameters.get('cavity_indices', [None])[0]):
                        wire.restoreGate(gate_tuple)
            self.redo_stack.append(last_action)
            self.generate_signal()

    def redo(self):
        if self.redo_stack:
            last_action = self.redo_stack.pop()
            action_type, gate_tuple = last_action
            gate, parameters, position, additional_graphics, gate_item, gate_scene_pos = gate_tuple
            if action_type == "add":
                for wire in self.scene.items():
                    if isinstance(wire, CircuitWire) and wire.index == (parameters.get('cavity_index') or parameters.get('cavity_indices', [None])[0]):
                        wire.restoreGate(gate_tuple)
            elif action_type == "delete":
                for wire in self.scene.items():
                    if isinstance(wire, CircuitWire):
                        wire.deleteGate(gate_item)
            self.undo_stack.append(last_action)
            self.generate_signal()

    def delete_selected_gate(self):
        if self.selected_gate:
            for wire in self.scene.items():
                if isinstance(wire, CircuitWire):
                    gate_tuple = wire.deleteGate(self.selected_gate)
                    if gate_tuple:
                        self.undo_stack.append(("delete", gate_tuple))
            self.selected_gate = None
            self.generate_signal()

    def clear_all(self):
        self.undo_stack.append(("clear", [(gate, params, pos, graphics, gate_item, gate_scene_pos) 
                                          for wire in self.scene.items() if isinstance(wire, CircuitWire) 
                                          for gate, params, pos, graphics, gate_item, gate_scene_pos in wire.gates]))
        self.redo_stack.clear()
        for wire in self.scene.items():
            if isinstance(wire, CircuitWire):
                wire.clear_all_gates()
        self.generate_signal()

    def edit_selected_gate(self):
        if self.selected_gate:
            for wire in self.scene.items():
                if isinstance(wire, CircuitWire):
                    for gate, parameters, position, additional_graphics, gate_item, gate_scene_pos in wire.gates:
                        if gate_item == self.selected_gate:
                            parent_widget = self.scene.views()[0]
                            dialog = GateParameterDialog(gate, parameters['t_i'], self.chi, self.t1, self.Nc, parent_widget, edit_mode=True)
                            dialog.start_time_input.setText(str(parameters['t_i']))
                            dialog.phase_input.setText(str(parameters['phase']))
                            
                            if gate == 'SNAP':
                                dialog.thetas_input.setText(','.join(map(str, parameters['thetas'])))
                                dialog.length_factor_input.setText(str(parameters['delta_t'] * self.chi / (2 * np.pi)))
                                if self.Nc > 1:
                                    dialog.cavity_indices_input.setText(','.join(map(str, parameters['cavity_indices'])))
                            elif gate == 'Displacement':
                                dialog.alpha_input.setText(str(parameters['alpha']))
                                dialog.length_factor_input.setText(str(parameters['delta_t'] * self.chi / (2 * np.pi)))
                                if self.Nc > 1:
                                    dialog.cavity_index_input.setText(str(parameters['cavity_index']))
                            elif gate == 'ECD':
                                dialog.beta_input.setText(str(parameters['beta']))
                                dialog.unit_amp_input.setText(str(parameters['unit_amp']))
                                if self.Nc > 1:
                                    dialog.cavity_index_input.setText(str(parameters['cavity_index']))
                            elif gate == 'Rotation':
                                dialog.theta_input.setText(str(parameters['theta']))
                                dialog.phi_input.setText(str(parameters['phi']))
                                dialog.length_factor_input.setText(str(parameters['delta_t'] * self.chi / (2 * np.pi)))
                                dialog.unit_amp_input.setText(str(parameters['unit_amp']))
                            
                            if dialog.exec_() == QDialog.Accepted:
                                try:
                                    new_params = dialog.get_parameters()
                                except ValueError as e:
                                    print(e)
                                    return
                                parameters.update(new_params)
                                self.generate_signal()

def export_signal_to_csv(filename, tlist, transmon_I, transmon_Q, cavity_signals):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Transmon_I', 'Transmon_Q'] 
                        + [f'Cavity_{i+1}_I' for i in range(len(cavity_signals))] 
                        + [f'Cavity_{i+1}_Q' for i in range(len(cavity_signals))])
        for t, ti, tq, *cavities in zip(tlist, transmon_I, transmon_Q, *[item for sublist in cavity_signals for item in sublist]):
            writer.writerow([t, ti, tq] + list(cavities))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())