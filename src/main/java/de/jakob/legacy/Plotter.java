package de.jakob.legacy;

import de.jakob.DataPoint;
import de.jakob.Main;
import de.jakob.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.io.IOException;
import java.util.List;

import static de.jakob.Main.save;


public class Plotter extends JFrame {

    private final Plot plot;
    private boolean showPoints = true;
    private boolean showOnlyWrongPoints = true;
    private final NeuralNetwork nn;
    private final List<DataPoint> dataPoints;

    public Plotter(int width, int height, int pointSize, int precision, NeuralNetwork nn, List<DataPoint> dataPoints) {
        this.nn = nn;
        this.dataPoints = dataPoints;

        setTitle("Neural Network");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        plot = new Plot(width, height, pointSize, precision, nn, dataPoints, this);
        add(plot);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);

        createButtonPane(nn);
    }

    public void createButtonPane(NeuralNetwork nn) {
        JFrame buttonFrame = new JFrame("Buttons");
        buttonFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        buttonFrame.add(new ButtonPane(nn, this));
        buttonFrame.pack();
        buttonFrame.setVisible(true);
    }

    public void refresh() {
        double cost = nn.totalCost(dataPoints);
        System.out.println("Cost: " + cost);

        plot.visualize(plot.getGraphics());
        if (showPoints)
            plot.drawPoints(plot.getGraphics());
    }

    public static class ButtonPane extends JPanel {
        private final Plotter plotter;

        public ButtonPane(NeuralNetwork nn, Plotter plotter) {
            this.plotter = plotter;
            setPreferredSize(new Dimension(500, 500));
            setBackground(new Color(20, 20, 20));

            add(createSaveButton(nn));
            add(createRefreshButton());
            add(createTogglePointsButton());
            add(createToggleOnlyWrongPointsButton());
            add(createStartLearningButton(nn));
        }

        private JButton createToggleOnlyWrongPointsButton() {
            JButton toggleOnlyWrongPointsButton = new JButton("Toggle Only Wrong Points");
            toggleOnlyWrongPointsButton.addActionListener(e -> {
                plotter.showOnlyWrongPoints = !plotter.showOnlyWrongPoints;
//                if (Main.isIsStillLearning() && Main.isLearning()) {
//                    //Main.refreshAfterIteration();
//                } else {
//                    plotter.refresh();
//                }
            });
            return createBasicButton(toggleOnlyWrongPointsButton);
        }

        private JButton createRefreshButton() {
            JButton refreshButton = new JButton("Refresh");
            refreshButton.addActionListener(e -> {
//                if (Main.isIsStillLearning() && Main.isLearning()) {
//                    //Main.refreshAfterIteration();
//                } else {
//                    plotter.refresh();
//                }
            });
            return createBasicButton(refreshButton);
        }

        private JButton createTogglePointsButton() {
            JButton togglePointsButton = new JButton("Toggle Points");
            togglePointsButton.addActionListener(e -> {
                plotter.showPoints = !plotter.showPoints;
//                if (Main.isIsStillLearning() && Main.isLearning()) {
//                    //Main.refreshAfterIteration();
//                } else {
//                    plotter.refresh();
//                }
            });
            return createBasicButton(togglePointsButton);
        }

        private JButton createStartLearningButton(NeuralNetwork nn) {
            JButton startLearningButton = new JButton("Start Learning");
            startLearningButton.addActionListener(e -> {
                //Main.startLearning();
            });
            return createBasicButton(startLearningButton);
        }

        public JButton createSaveButton(NeuralNetwork nn) {
            JButton saveButton = new JButton("Save");
            saveButton.addActionListener(e -> {
                save(nn);
            });
            return createBasicButton(saveButton);
        }

        private JButton createBasicButton(JButton saveButton) {
            saveButton.setBackground(new Color(20, 20, 20));
            saveButton.setForeground(new Color(255, 255, 255));
            saveButton.setBorder(BorderFactory.createLineBorder(new Color(255, 255, 255)));
            saveButton.setPreferredSize(new Dimension(200, 100));
            return saveButton;
        }


    }

    public static class Plot extends JPanel {

        private final int width, height;
        private final int pointSize, precision;

        private final Plotter plotter;
        private final NeuralNetwork nn;
        private final List<DataPoint> dataPoints;

        private final Color redPoint, bluePoint;
        private final Color red, blue;

        public Plot(int width, int height, int pointSize, int precision, NeuralNetwork nn, List<DataPoint> dataPoints, Plotter plotter) {
            this.width = width;
            this.height = height;
            this.pointSize = pointSize;
            this.precision = precision;
            this.nn = nn;
            this.dataPoints = dataPoints;
            this.plotter = plotter;

            redPoint = new Color(255, 81, 81);
            bluePoint = new Color(81, 81, 255);

            red = new Color(255, 81, 81, 50);
            blue = new Color(81, 81, 255, 50);

            setLayout(null);
            setPreferredSize(new Dimension(width, height));
            setBackground(new Color(20, 20, 20));
        }

        @Override
        public void paintComponent(java.awt.Graphics g) {
            super.paintComponent(g);
            System.out.println("Painting");

            visualize(g);
            drawPoints(g);
        }


        public void drawPoints(Graphics g) {
            for(DataPoint point : dataPoints) {
                if(plotter.showOnlyWrongPoints && point.expectedOutputs()[nn.classify(point.inputs())] == 1)
                    continue;

                int x = (int) (point.inputs()[0] * width);
                int y = (int) (point.inputs()[1] * height);

                if(point.expectedOutputs()[0] == 0) {
                    g.setColor(bluePoint);
                } else {
                    g.setColor(redPoint);
                }

                g.fillOval(x - pointSize / 2, y - pointSize / 2, pointSize, pointSize);
            }
        }

        public void visualize(Graphics g) {
            g.setColor(new Color(20, 20, 20));
            g.fillRect(0, 0, width, height);

            for(int x = 0; x < width; x += precision) {
                for(int y = 0; y < height; y += precision) {
                    double[] input = new double[] {x / (double) width, y / (double) height};
                    int output = nn.classify(input);

                    int[] colorValues;

                    if(output == 0) {
                        colorValues = new int[] {red.getRed(), red.getGreen(), red.getBlue()};
                    } else {
                        colorValues = new int[] {blue.getRed(), blue.getGreen(), blue.getBlue()};
                    }

                    int alpha = (int) (nn.calculate(input)[output] * 100);

                    Color color = new Color(colorValues[0], colorValues[1], colorValues[2], alpha);
                    g.setColor(color);

                    g.fillRect(x, y, precision, precision);
                }
            }
        }
    }
}
