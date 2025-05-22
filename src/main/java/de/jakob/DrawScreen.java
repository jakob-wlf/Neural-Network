package de.jakob;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;

public class DrawScreen extends JFrame {
    private static final int PIXEL_SIZE = 10;
    private static final int GRID_SIZE = Main.size;
    private static final int SCREEN_SIZE = PIXEL_SIZE * GRID_SIZE;

    private final BufferedImage canvas;
    private final DrawPanel panel;
    private final JPanel predictionPanel;
    private final NeuralNetwork nn;
    private final JLabel[] predictionLabels;

    private final String[] categories = {
            "airplane", "alarm clock", "bear", "axe", "bridge",
            "windmill", "telephone", "house", "butterfly", "tree"
    };

    public DrawScreen(NeuralNetwork nn) {
        this.nn = nn;

        setTitle("Neural Network Drawing");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(10, 10));

        canvas = new BufferedImage(GRID_SIZE, GRID_SIZE, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D cg = canvas.createGraphics();
        cg.setColor(Color.BLACK);
        cg.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
        cg.dispose();

        panel = new DrawPanel();
        panel.setPreferredSize(new Dimension(SCREEN_SIZE, SCREEN_SIZE));
        panel.setBorder(BorderFactory.createLineBorder(Color.DARK_GRAY, 4));
        add(panel, BorderLayout.CENTER);

        predictionPanel = new JPanel();
        predictionPanel.setLayout(new BoxLayout(predictionPanel, BoxLayout.Y_AXIS));
        predictionLabels = new JLabel[categories.length];
        for (int i = 0; i < categories.length; i++) {
            predictionLabels[i] = new JLabel();
            predictionLabels[i].setFont(new Font("Monospaced", Font.BOLD, 20));
            predictionPanel.add(predictionLabels[i]);
        }

        JScrollPane scrollPane = new JScrollPane(predictionPanel);
        scrollPane.setPreferredSize(new Dimension(300, SCREEN_SIZE));
        add(scrollPane, BorderLayout.EAST);

        JPanel controlPanel = new JPanel();
        controlPanel.setLayout(new FlowLayout(FlowLayout.CENTER, 15, 10));
        controlPanel.setBackground(new Color(0xF0F0F0));

        JButton stopButton = createStyledButton("Stop Learning");
        stopButton.addActionListener(e -> Main.stopLearning());

        JButton clearButton = createStyledButton("Clear");
        clearButton.addActionListener(e -> clearCanvas());

        JButton randomImageButton = createStyledButton("Random Image");
        randomImageButton.addActionListener(e -> showRandom(nn, Main.dataPoints));

        JButton randomValidationImageButton = createStyledButton("Random Validation");
        randomValidationImageButton.addActionListener(e -> showRandom(nn, Main.validationDataPoints));

        JButton startLearningButton = createStyledButton("Start Learning");
        startLearningButton.addActionListener(e -> Main.learning = true);

        controlPanel.add(stopButton);
        controlPanel.add(clearButton);
        controlPanel.add(randomImageButton);
        controlPanel.add(randomValidationImageButton);
        controlPanel.add(startLearningButton);

        add(controlPanel, BorderLayout.SOUTH);

        pack();
        setResizable(false);
        setLocationRelativeTo(null);
        setVisible(true);

        updatePredictionLabels(); // Initialize with blank canvas
    }

    private JButton createStyledButton(String text) {
        JButton button = new JButton(text);
        button.setFont(new Font("SansSerif", Font.BOLD, 14));
        button.setBackground(new Color(0xFFFFFF));
        button.setFocusPainted(false);
        button.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(new Color(0xCCCCCC)),
                BorderFactory.createEmptyBorder(5, 15, 5, 15)
        ));
        return button;
    }

    private void clearCanvas() {
        Graphics2D g = canvas.createGraphics();
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, GRID_SIZE, GRID_SIZE);
        g.dispose();
        panel.repaint();
        updatePredictionLabels();
    }

    private void showRandom(NeuralNetwork nn, List<DataPoint> list) {
        int randomIndex = (int) (Math.random() * list.size());
        DataPoint dp = list.get(randomIndex);
        displayImage(dp.inputs());
    }

    public double[] getDrawingData() {
        double[] data = new double[GRID_SIZE * GRID_SIZE];
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                int gray = canvas.getRGB(x, y) & 0xFF;
                data[y * GRID_SIZE + x] = gray / 255.0;
            }
        }
        return data;
    }

    public void displayImage(double[] data) {
        if (data.length != GRID_SIZE * GRID_SIZE) {
            throw new IllegalArgumentException("Data array must be of length " + GRID_SIZE * GRID_SIZE);
        }
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                int gray = (int) (Math.max(0, Math.min(255, data[y * GRID_SIZE + x] * 255)));
                canvas.setRGB(x, y, new Color(gray, gray, gray).getRGB());
            }
        }
        panel.repaint();
        updatePredictionLabels();
    }

    private void updatePredictionLabels() {
        double[] data = getDrawingData();
        double[] outputs = nn.calculate(data);

        Integer[] indices = new Integer[categories.length];
        for (int i = 0; i < indices.length; i++) indices[i] = i;

        Arrays.sort(indices, (a, b) -> Double.compare(outputs[b], outputs[a]));

        for (int i = 0; i < categories.length; i++) {
            int idx = indices[i];
            predictionLabels[i].setText(String.format("%-15s: %.2f", categories[idx], outputs[idx]));
        }
    }

    private class DrawPanel extends JPanel {
        private boolean drawing = false;

        public DrawPanel() {
            setBackground(Color.BLACK);
            addMouseListener(new MouseAdapter() {
                @Override public void mousePressed(MouseEvent e) { drawing = true; drawAt(e.getX(), e.getY()); }
                @Override public void mouseReleased(MouseEvent e) { drawing = false; }
            });
            addMouseMotionListener(new MouseMotionAdapter() {
                @Override public void mouseDragged(MouseEvent e) {
                    if (drawing) {
                        drawAt(e.getX(), e.getY());
                        updatePredictionLabels(); // Recalculate on each stroke
                    }
                }
            });
        }

        private void drawAt(int x, int y) {
            int gx = x / PIXEL_SIZE;
            int gy = y / PIXEL_SIZE;
            if (gx >= 0 && gx < GRID_SIZE && gy >= 0 && gy < GRID_SIZE) {
                canvas.setRGB(gx, gy, Color.WHITE.getRGB());
                repaint(gx * PIXEL_SIZE, gy * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
            }
        }

        @Override protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            for (int y = 0; y < GRID_SIZE; y++) {
                for (int x = 0; x < GRID_SIZE; x++) {
                    int gray = canvas.getRGB(x, y) & 0xFF;
                    g.setColor(new Color(gray, gray, gray));
                    g.fillRect(x * PIXEL_SIZE, y * PIXEL_SIZE, PIXEL_SIZE, PIXEL_SIZE);
                }
            }
        }
    }
}

