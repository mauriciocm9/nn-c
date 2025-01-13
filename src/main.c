#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mdarray.h"
#include "image.h"

#define IMG_SIZE 784

int read_int(FILE* file) {
    unsigned char msb[4];
    if(fread(msb, 1, 4, file) < 1) return -1;

    int out = 0;
    for (size_t i = 0; i <4; i++) {
        out = (out << 8) | msb[i];
    }
    return out;
}

int read_images(char* filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    int msb = read_int(file); 
    if(msb == -1) return -1;
    int n_samples = read_int(file); 
    if(n_samples == -1) return -1;

    int r_size = read_int(file); 
    if(r_size == -1) return -1;
    int c_size = read_int(file); 
    if(c_size == -1) return -1;


    printf("%d\n", msb);
    printf("%d\n", n_samples);
    printf("%d\n", r_size);
    printf("%d\n", c_size);

    unsigned char imgbytes[IMG_SIZE];
    size_t bytes_read;

    MDArray* imgs = malloc(sizeof(MDArray)*n_samples);
    int index = 0;

    while ((bytes_read = fread(imgbytes, 1, IMG_SIZE, file)) > 0) {
        size_t shape[] = {28, 28};
        MDArray* img_arr = mdarray_create(2, shape, sizeof(int));
        for (size_t i = 0; i < bytes_read; i++) {
            size_t indices[] = {i/28, i%28};
            mdarray_set_element(img_arr, indices, &imgbytes[i]);
        }
        memcpy(imgs + (index * sizeof(MDArray)), img_arr, sizeof(MDArray));
        index++;

        //size_t x[] = {12, 12};
        //printf("%d\n", *(unsigned char*)mdarray_get_element(img_arr, x));
    }

    fclose(file);
    return 0;
}

int read_labels(char* filename) {
    return 1;
}

int main() {
    int x,y;
    x = read_images("../data/train-images.idx3-ubyte");
    //y = read("../data/train-labels.idx1-ubyte");
    printf("Hello world, %d %d\n", x, y);
}
