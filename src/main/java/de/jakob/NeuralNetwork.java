package de.jakob;


import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.FileReader;
import java.util.List;

// NeuralNetwork.java
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import java.io.FileReader;
import java.util.List;
import java.util.Random;

public class NeuralNetwork {
    private Layer[] layers;

    public NeuralNetwork(int... nodes) {
        createNetwork(nodes);
    }

    public NeuralNetwork() {
        load();
    }

    private void createNetwork(int... nodes) {
        layers = new Layer[nodes.length - 1];
        for (int i = 0; i < layers.length; i++) {
            layers[i] = new Layer(nodes[i], nodes[i + 1]);
        }
    }

    @SuppressWarnings("unchecked")
    public void load() {
        JSONParser parser = new JSONParser();
        try (FileReader reader = new FileReader("D:\\Jakob\\Programming\\Java\\NeuralNetwork\\src\\main\\resources\\neural_network.json")) {
            JSONObject jsonObject = (JSONObject) parser.parse(reader);
            int count = jsonObject.size();
            layers = new Layer[count];
            for (int i = 0; i < count; i++) {
                JSONObject layerObj = (JSONObject) jsonObject.get("layer_" + i);
                int numIn  = ((Long) layerObj.get("nIn")).intValue();
                int numOut = ((Long) layerObj.get("nOut")).intValue();
                Layer layer = new Layer(numIn, numOut);
                layer.loadFromJson(layerObj);
                layers[i] = layer;
            }
        } catch (Exception e) {
            System.out.println("Failed to load network, creating random one.");
            createNetwork(Main.size * Main.size, 256, 256, 128, 10);
        }

        System.out.println("initial cost: " + totalCost(Main.getRandomPoints(Main.dataPoints, 250)));
    }


    /** Forward pass */
    public double[] calculate(double[] input) {
        double[] output = input;
        for (Layer layer : layers) {
            output = layer.calculateOutputs(output);
        }
        return output;
    }

    /** Cross‑entropy cost for one point */
    public double cost(DataPoint dataPoint) {
        double[] a = calculate(dataPoint.inputs());
        double sum = 0;
        double eps = 1e-12;
        double[] y = dataPoint.expectedOutputs();

        for (int i = 0; i < a.length; i++) {
            // Ensure a[i] and 1 - a[i] are not too close to 0
            double a_i = Math.max(Math.min(a[i], 1 - eps), eps); // Clamp to avoid values too close to 0 or 1
            double one_minus_a_i = Math.max(Math.min(1 - a_i, 1 - eps), eps);

            sum += - (y[i] * Math.log(a_i) + (1 - y[i]) * Math.log(one_minus_a_i));
        }

        return sum;
    }


    /** Single mini‑batch gradient step */
    public void learn(List<DataPoint> dataPoints, double learningRate) {
        // reset any old gradients
        clearAllGradients();

        for (DataPoint dp : dataPoints) {
            updateAllGradients(dp);
        }


        // apply the average gradient
        applyAllGradients(learningRate / dataPoints.size());
    }


    private void updateAllGradients(DataPoint dp) {
        // forward‑prop
        calculate(dp.inputs());

        // back‑prop on output layer
        Layer out = layers[layers.length - 1];
        double[] nodeVals = out.calculateOutputLayerNodeValues(dp.expectedOutputs());
        out.updateGradients(nodeVals);

        // back‑prop through hidden
        for (int i = layers.length - 2; i >= 0; i--) {
            Layer cur = layers[i];
            Layer next = layers[i + 1];
            nodeVals = cur.calculateHiddenLayerNodeValues(next, nodeVals);
            cur.updateGradients(nodeVals);
        }
    }

    private void applyAllGradients(double lr) {
        for (Layer layer : layers) {
            layer.applyGradients(lr);
        }
    }

    private void clearAllGradients() {
        for (Layer layer : layers) {
            layer.clearGradients();
        }
    }

    /** Average cost over a dataset */
    public double totalCost(List<DataPoint> data) {
        double sum = 0;
        for (DataPoint dp : data) sum += cost(dp);
        return sum / data.size();
    }

    /** How many are classified correctly */
    public int correctPoints(List<DataPoint> data) {
        int c = 0;
        for (DataPoint dp : data) {
            if (dp.expectedOutputs()[ classify(dp.inputs()) ] == 1) c++;
        }
        return c;
    }

    /** argmax of the network’s output */
    public int classify(double[] input) {
        double[] out = calculate(input);
        int max = 0;
        for (int i = 1; i < out.length; i++) {
            if (out[i] > out[max]) max = i;
        }
        return max;
    }

    public Layer[] getLayers() {
        return layers;
    }
}
