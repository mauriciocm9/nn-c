#include <stdio.h>
#include <jpeglib.h>
#include <math.h>
#include "mdarray.h"
#include "linear.h"

#define IMG_SIZE 784

void write_jpeg(MDArray* imgs) {
    unsigned char grayscale_data[IMG_SIZE];
    for (size_t x = 0; x < 28; x++) {
        for (size_t y = 0; y < 28; y++) {
            size_t idx[] = {0, x, y};
            grayscale_data[x * 28 + y] = (unsigned char)*(double*)mdarray_get_element(imgs, idx);
        }
    }

    // Set up libjpeg structures
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    // Open the output file
    FILE *outfile = fopen("example.jpeg", "wb");
    if (!outfile) {
        perror("Error opening file");
        return;
    }
    jpeg_stdio_dest(&cinfo, outfile);

    // Set image parameters
    cinfo.image_width = 28;       // Image width
    cinfo.image_height = 28;      // Image height
    cinfo.input_components = 1;   // Grayscale (1 channel)
    cinfo.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 100, TRUE); // Quality: 0-100
    jpeg_start_compress(&cinfo, TRUE);

    // Write the grayscale image row by row
    while (cinfo.next_scanline < cinfo.image_height) {
        JSAMPROW row_pointer[1];
        row_pointer[0] = &grayscale_data[cinfo.next_scanline * 28];
        //size_t idx[] = {0, cinfo.next_scanline, 0};
        //row_pointer[0] = (unsigned char*)mdarray_get_element(imgs, idx);
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    // Finish compression and cleanup
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);
}

int read_int(FILE* file) {
    unsigned char msb[4];
    if(fread(msb, 1, 4, file) < 1) return -1;

    int out = 0;
    for (size_t i = 0; i <4; i++) {
        out = (out << 8) | msb[i];
    }
    return out;
}

MDArray* read_images(char* filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    int msb = read_int(file);
    if(msb == -1) return NULL;
    int n_samples = read_int(file);
    if(n_samples == -1) return NULL;

    int r_size = read_int(file);
    if(r_size == -1) return NULL;
    int c_size = read_int(file);
    if(c_size == -1) return NULL;

    printf("Magic number: %d\n", msb);
    printf("Number images: %d\n", n_samples);
    printf("Image shape %dx%d\n", r_size, c_size);

    unsigned char imgbytes[IMG_SIZE];
    size_t bytes_read;

    size_t shape[] = {n_samples, r_size, c_size};
    MDArray* imgs = mdarray_create(3, shape, sizeof(double));
    int index = 0;

    while ((bytes_read = fread(imgbytes, 1, IMG_SIZE, file)) > 0) {
        for (size_t i = 0; i < bytes_read; i++) {
            size_t indices[] = {index, i/28, i%28};
            unsigned char c = imgbytes[i];
            double x = (double)c;
            mdarray_set_element(imgs, indices, &x);
        }
        index++;
    }

    write_jpeg(imgs);
    fclose(file);

    return imgs;
}

MDArray* read_labels(char* filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return NULL;
    }

    int msb = read_int(file);
    if(msb == -1) return NULL;
    int n_samples = read_int(file);
    if(n_samples == -1) return NULL;

    printf("Magic number: %d\n", msb);
    printf("Number labels: %d\n", n_samples);

    size_t shape[] = {n_samples};
    MDArray* labels = mdarray_create(1, shape, sizeof(double));

    unsigned char* labelbytes = (unsigned char*)malloc(sizeof(unsigned char) * n_samples);
    size_t bytes_read;

    while ((bytes_read = fread(labelbytes, 1, n_samples, file)) > 0) {
        for (size_t i = 0; i < bytes_read; i++) {
            size_t indices[] = {i};
            unsigned char c = labelbytes[i];
            double x = (double)c;
            mdarray_set_element(labels, indices, &x);
        }
    }

    size_t indices[] = {0};
    printf("First label is: %f\n", *(double*)mdarray_get_element(labels, indices));
    return labels;
}

int main() {
    MDArray* images = read_images("../data/train-images.idx3-ubyte");
    MDArray* labels = read_labels("../data/train-labels.idx1-ubyte");


    LinearModel* model = linearmodel_new(images, labels);
    MDArray* out = linearmodel_forward(model);
    size_t shape[] = {0,0};
    printf("Value is %f\n", *(double*) mdarray_get_element(out, shape));
}
