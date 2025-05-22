package de.jakob.preprocessing;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class DoodleCsvGenerator {

    private static final int IMG_SIZE = 80;
    private static final File DOODLE_FOLDER = new File("\\D:\\Jakob\\Programming\\Datasets\\Doodles");
    private static final File OUTPUT_CSV = new File("doodles_" + IMG_SIZE + "px.csv");

    private static final boolean clampValues = true;

    public static void main(String[] args) throws IOException {
        List<File> labelFolders = Arrays.asList(Objects.requireNonNull(DOODLE_FOLDER.listFiles(File::isDirectory)));
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(OUTPUT_CSV))) {
            for (int labelIndex = 0; labelIndex < labelFolders.size(); labelIndex++) {
                File folder = labelFolders.get(labelIndex);
                File[] images = folder.listFiles((dir, name) -> name.toLowerCase().endsWith(".png"));

                if (images == null) continue;

                for (File imageFile : images) {
                    float[] pixels = extractPixels(imageFile);
                    float[] labelVector = oneHot(labelIndex, labelFolders.size());

                    StringBuilder sb = new StringBuilder();
                    for (float l : labelVector) sb.append((int) l).append(",");
                    for (int i = 0; i < pixels.length; i++) {
                        sb.append(pixels[i]);
                        if (i < pixels.length - 1) sb.append(",");
                    }
                    writer.write(sb.toString());
                    writer.newLine();
                }

                System.out.println("Processed folder: " + folder.getName());
            }
        }

        System.out.println("CSV generation complete: " + OUTPUT_CSV.getAbsolutePath());
    }

    private static float[] extractPixels(File imageFile) throws IOException {
        BufferedImage original = ImageIO.read(imageFile);

        BufferedImage resized = new BufferedImage(
                IMG_SIZE, IMG_SIZE,
                BufferedImage.TYPE_BYTE_GRAY
        );
        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
        g2d.drawImage(original, 0, 0, IMG_SIZE, IMG_SIZE, null);
        g2d.dispose();

        Raster raster = resized.getRaster();
        float[] pixels = new float[IMG_SIZE * IMG_SIZE];
        for (int y = 0; y < IMG_SIZE; y++) {
            for (int x = 0; x < IMG_SIZE; x++) {
                int gray = raster.getSample(x, y, 0);

                pixels[y * IMG_SIZE + x] = clampValues ? gray < 255 ? 1 : 0 : 1 - (gray / 255.0f);
            }
        }

        return pixels;
    }



    private static float[] oneHot(int index, int total) {
        float[] vector = new float[total];
        vector[index] = 1.0f;
        return vector;
    }
}
