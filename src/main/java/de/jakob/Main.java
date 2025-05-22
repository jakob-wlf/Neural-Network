package de.jakob;

import de.jakob.legacy.Plotter;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import java.io.*;
import java.sql.SQLOutput;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class Main {

    public static List<DataPoint> dataPoints;
    public static List<DataPoint> validationDataPoints;

    public static final int size = 80;

   public static void main(String[] args) throws InterruptedException {
       System.out.println("Runnning Neural network in Java");
       loadDataPoints();
       System.out.println("Creating neural network");
       NeuralNetwork nn = new NeuralNetwork();
       new DrawScreen(nn);
       learn(nn);
    }

    private static void loadDataPoints() {
        System.out.println("Loading Data points...");
        dataPoints = getRandomPoints(getPointsAsList(), 16000);
        validationDataPoints = getPointsAsList();
        validationDataPoints.removeIf(dataPoints::contains);
        System.out.println("Finished loading data points");
    }

    private static boolean isStillLearning = true;
    public static boolean learning = false;

    public static void learn(NeuralNetwork nn) throws InterruptedException {
        System.out.println("Starting learning");
        isStillLearning = true;
        long startTime = System.currentTimeMillis();

        List<DataPoint> randomBatch = getRandomPoints(dataPoints, 250);

        double lowest_cost = nn.totalCost(randomBatch);

        int batchSize = 32;
        long iteration = 0;
        int epoch = 0;

        double initialLR = 0.05;
        double decayRate = 0.99;

        Random rand = new Random();

        while(isStillLearning) {
            if(!learning) {
                Thread.sleep(1000);
                continue;
            }

            List<List<DataPoint>> batches = getBatches(dataPoints, batchSize);
            if(epoch % 4 == 0)
                randomBatch = getRandomPoints(dataPoints, 250);

            epoch++;

            for(List<DataPoint> batch : batches) {
                float learningRate = (float) (initialLR * Math.pow(decayRate, epoch));
                List<DataPoint> augmentedBatch = new ArrayList<>();

                for(DataPoint dp : batch) {
                    if(rand.nextDouble() < 0.3) {
                        DataPoint augmented = augment(dp);
                        augmentedBatch.add(augmented);
                    } else {
                        augmentedBatch.add(dp);
                    }
                }

                nn.learn(augmentedBatch, learningRate);
                iteration++;

                if (iteration % 200 == 0) {
                    double cost = nn.totalCost(randomBatch);
                    System.out.println("Step: " + iteration);
                    System.out.println("Cost: " + cost);
                    System.out.println("Lowest Cost: " + lowest_cost);
                    System.out.println("Learning Rate: " + learningRate);

                    if (cost < lowest_cost) {
                        lowest_cost = cost;
                        save(nn);
                        System.out.println("Finished saving");
                    }

                    if(cost < 0.001) {
                        System.out.println("Stopping training, cost is low enough.");
                        System.out.println("Final cost: " + cost);

                        System.out.println("Time taken: " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
                        isStillLearning = false;
                    }
                }

                if(!isStillLearning) {
                    break;
                }
            }
        }

        double cost = nn.totalCost(randomBatch);

        if (cost < lowest_cost) {
            save(nn);
            System.out.println("Finished saving");
        }
        else {
            nn.load();
        }

        System.out.println("Final cost: " + cost);
        System.out.println("Step: " + iteration);
        System.out.println("Lowest Cost: " + lowest_cost);
    }

    public static void stopLearning() {
        isStillLearning = false;
    }


    @SuppressWarnings("unchecked")
    public static void save(NeuralNetwork nn) {
        System.out.println("Saving...");

        JSONObject obj = new JSONObject();
        Layer[] layers = nn.getLayers();

        for (int i = 0; i < layers.length; i++) {
            Layer layer = layers[i];
            JSONObject layerObj = new JSONObject();

            // Input/output dimensions
            layerObj.put("nIn", layer.getnIn());
            layerObj.put("nOut", layer.getnOut());

            // Serialize weights as a 2D array
            double[][] weights = layer.getWeights();
            JSONArray weightsArray = new JSONArray();
            for (double[] row : weights) {
                JSONArray rowArray = new JSONArray();
                for (double val : row) {
                    rowArray.add(val);
                }
                weightsArray.add(rowArray);
            }
            layerObj.put("weights", weightsArray);

            // Serialize biases as an array
            double[] biases = layer.getBiases();
            JSONArray biasesArray = new JSONArray();
            for (double bias : biases) {
                biasesArray.add(bias);
            }
            layerObj.put("biases", biasesArray);

            obj.put("layer_" + i, layerObj);
        }

        try (FileWriter file = new FileWriter("D:\\Jakob\\Programming\\Java\\NeuralNetwork\\src\\main\\resources\\neural_network.json")) {
            file.write(obj.toJSONString());
            file.flush();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }




    public static List<DataPoint> getRandomPoints(List<DataPoint> dataPoints, int amount) {
        List<DataPoint> randomPoints = new ArrayList<>();
        for (int i = 0; i < amount; i++) {
            randomPoints.add(dataPoints.get((int) (Math.random() * dataPoints.size())));
        }
        return randomPoints;

    }

    private static List<List<DataPoint>> getBatches(List<DataPoint> data, int batchSize) {
        List<List<DataPoint>> batches = new ArrayList<>();
        Collections.shuffle(data, ThreadLocalRandom.current()); // Shuffle once per epoch

        for (int i = 0; i < data.size(); i += batchSize) {
            int end = Math.min(i + batchSize, data.size());
            batches.add(data.subList(i, end));
        }

        return batches;
    }

    public static DataPoint augment(DataPoint dp) {
        int size = 80;
        double[][] grid = new double[size][size];

        // Reshape flat input
        for (int i = 0; i < size; i++)
            System.arraycopy(dp.inputs(), i * size, grid[i], 0, size);

        Random rand = new Random();

        int maxShift = rand.nextDouble() < 0.1 ? rand.nextBoolean() ? 3 : 2 : 1;
        int dx = rand.nextInt(2 * maxShift + 1) - maxShift;  // -2 to +2
        int dy = rand.nextInt(2 * maxShift + 1) - maxShift;

        double[][] shifted = new double[size][size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                int srcI = i - dy;
                int srcJ = j - dx;
                if (srcI >= 0 && srcI < size && srcJ >= 0 && srcJ < size)
                    shifted[i][j] = grid[srcI][srcJ];
                else
                    shifted[i][j] = 0.0; // pad with white
            }
        }

        // MIRROR (horizontal flip) — rare
        if (rand.nextBoolean()) {
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size / 2; j++) {
                    double temp = shifted[i][j];
                    shifted[i][j] = shifted[i][size - 1 - j];
                    shifted[i][size - 1 - j] = temp;
                }
            }
        }

        // Add light pixel noise
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                if (rand.nextDouble() < 0.01)
                    shifted[i][j] = 1.0 - shifted[i][j];

        // Flatten to 1D again
        double[] augmented = new double[size * size];
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                augmented[i * size + j] = shifted[i][j];

        return new DataPoint(augmented, dp.expectedOutputs());
    }

    //TODO: pass length as parameter
    private static DataPoint convertToDataPoint(double[] values) {
        double[] inputs = new double[Main.size * Main.size];
        double[] expectedOutputs = new double[10];

        for (int i = 0; i < 10; i++) {
            expectedOutputs[i] = values[i];
        }

        //The rest of the values are the inputs
        for (int i = 10; i < values.length; i++) {
            inputs[i - 10] = values[i];
        }

        return new DataPoint(inputs, expectedOutputs);
    }

    private static List<DataPoint> getPointsAsList() {
        List<DataPoint> dataPoints = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader("D:\\Jakob\\Programming\\Java\\NeuralNetwork\\src\\main\\resources\\doodles_" + size + "px.csv"))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] current_values = line.split(",");
                double[] current_values_double = new double[current_values.length];

                for (int i = 0; i < current_values.length; i++) {
                    current_values_double[i] = Double.parseDouble(current_values[i]);
                }

                dataPoints.add(convertToDataPoint(current_values_double)); // Convert immediately
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return dataPoints;
    }


    public static boolean isIsStillLearning() {
        return isStillLearning;
    }
}