#include <stdlib.h>

typedef struct {
    int arr[28][28][3];
} Image;



Image* image_create() {
    Image* arr = (Image*)malloc(sizeof(Image));
    return arr;
}
