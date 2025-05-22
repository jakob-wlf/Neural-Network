package de.jakob;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.util.Random;
import java.util.Arrays;

public class Layer {
    private final int nIn, nOut;
    private final double[][] weights, gradW, velocityW;
    private final double[] biases,  gradB, velocityB;

    // once‑per‑layer buffers
    private final double[] inputs, zs, activations, nodeValsBuffer;

    private static final double MOMENTUM = 0.9;
    private static final Random RANDOM   = new Random();

    public Layer(int nIn, int nOut) {
        this.nIn  = nIn;
        this.nOut = nOut;

        weights    = new double[nOut][nIn];
        gradW      = new double[nOut][nIn];
        velocityW  = new double[nOut][nIn];


        biases     = new double[nOut];
        gradB      = new double[nOut];
        velocityB  = new double[nOut];

        inputs         = new double[nIn];
        zs             = new double[nOut];
        activations    = new double[nOut];
        nodeValsBuffer = new double[nOut];

        initRandomWeights();
    }

    /** Forward pass: z = W·in + b, then ReLU(z) */
    public double[] calculateOutputs(double[] in) {
        System.arraycopy(in, 0, inputs, 0, nIn);

        // 1) compute all z’s
        for (int j = 0; j < nOut; j++) {
            double z = biases[j];
            for (int i = 0; i < nIn; i++) {
                z += weights[j][i] * in[i];
            }
            zs[j] = z;
        }

        // 2) vectorized ReLU
        for (int j = 0; j < nOut; j++) {
            activations[j] = relu(zs[j]);
        }
        return activations;
    }

    /** dC/dz for output layer under cross‑entropy = (a - y) */
    public double[] calculateOutputLayerNodeValues(double[] expected) {
        for (int j = 0; j < nOut; j++) {
            nodeValsBuffer[j] = activations[j] - expected[j];
        }
        return nodeValsBuffer;
    }

    /** back‑prop into a hidden layer using ReLU′(z) */
    public double[] calculateHiddenLayerNodeValues(Layer next, double[] nextVals) {
        for (int j = 0; j < nOut; j++) {
            double sum = 0.0;
            for (int k = 0; k < next.nOut; k++) {
                sum += nextVals[k] * next.weights[k][j];
            }
            // ReLU′(z) = 1 if z>0, else 0
            nodeValsBuffer[j] = sum * (zs[j] > 0 ? 1.0 : 0.0);
        }
        return nodeValsBuffer;
    }

    /** Accumulate into gradW, gradB (unchanged) */
    public void updateGradients(double[] nodeVals) {
        for (int j = 0; j < nOut; j++) {
            gradB[j] += nodeVals[j];
            for (int i = 0; i < nIn; i++) {
                gradW[j][i] += inputs[i] * nodeVals[j];
            }
        }
    }

    /** Apply (and zero) gradients with momentum (unchanged) */
    public void applyGradients(double lr) {
        for (int j = 0; j < nOut; j++) {
            velocityB[j] = MOMENTUM * velocityB[j] + lr * gradB[j];
            biases[j] -= velocityB[j];

            for (int i = 0; i < nIn; i++) {
                velocityW[j][i] = MOMENTUM * velocityW[j][i] + lr * gradW[j][i];
                weights[j][i] -= velocityW[j][i];
            }
        }
    }

    /** Zero‑out gradients (unchanged) */
    public void clearGradients() {
        Arrays.fill(gradB, 0.0);
        for (int j = 0; j < nOut; j++) {
            Arrays.fill(gradW[j], 0.0);
        }

    }

    private void initRandomWeights() {
        double scale = Math.sqrt(2.0 / nIn);
        for (int j = 0; j < nOut; j++) {
            for (int i = 0; i < nIn; i++) {
                weights[j][i] = RANDOM.nextGaussian() * scale;
            }
        }

    }

    private static double relu(double x) {
        return x > 0 ? x : 0;
    }


    /** JSON loader adapted for flattened storage */
    public void loadFromJson(JSONObject layerObj) {
        JSONArray weightsArray = (JSONArray) layerObj.get("weights");
        for (int j = 0; j < nOut; j++) {
            JSONArray row = (JSONArray) weightsArray.get(j);
            for (int i = 0; i < nIn; i++) {
                weights[j][i] = ((Number) row.get(i)).doubleValue();
            }
        }

        JSONArray biasesArray = (JSONArray) layerObj.get("biases");
        for (int j = 0; j < nOut; j++) {
            biases[j] = ((Number) biasesArray.get(j)).doubleValue();
        }
    }


    public int getnIn() {
        return nIn;
    }

    public int getnOut() {
        return nOut;
    }

    public double[][] getWeights() {
        return weights;
    }

    public double[] getBiases() {
        return biases;
    }
}
