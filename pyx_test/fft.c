#include <math.h>
#include <stdio.h>
#include <assert.h>

inline void zmult(double *r, double *i, double r0, double i0, double r1, double i1) {
    *r = (r0 * r1 - i0 * i1);
    *i = (r0 * i1 + r1 * i0);
}

size_t _fft2(double *x, size_t N, int s, double *r, double *i) {
    static double *r_0 = NULL;
    static double *i_0 = NULL;
    static double *x_0 = NULL;
    if(s == 1) {
        r_0 = r;
        i_0 = i;
        x_0 = x;
    }
    if(N==1) {
        r[0] = x[0];
    } else {
        //even
        _fft2(x, N/2, 2*s, r,i);
        //odd
        _fft2(x+s, N/2, 2*s, r+N/2, i+N/2);
        for(int k=0;k<N/2;k++) {
            double tr = r[k];
            double ti = i[k];

            double rcos = cos(2.0 * M_PI * (double)k / (double)N);
            double isin = -1 * sin(2.0 * M_PI * (double)k / (double)N);

            r[k] = tr + (rcos * r[k+N/2] - isin * i[k+N/2]);
            i[k] = ti + (rcos * i[k+N/2] + isin * r[k+N/2]);

            r[k+(N/2)] = tr - (rcos * r[k+N/2] - isin * i[k+N/2]);
            i[k+(N/2)] = ti - (rcos * i[k+N/2] + isin * r[k+N/2]);
        }
    }
    return N;
}


size_t fft2(size_t N, double *x, double *r, double *i) {
    assert(N % 2 == 0);
    return _fft2(x, N, 1, r, i);
}

void print_array(double *a, size_t N) {
    int i=0;
    for(i=0;i<N-1;i++) {
        printf("%.2f, ", a[i]);
    }
    printf("%.2f", a[i]);
}


int main(int argc, char *argv[]) {
    double x[] = {0, 1, 2, 3, 4, 5, 6, 7};
    double r[] = {0, 0, 0, 0, 0, 0, 0, 0};
    double i[] = {0, 0, 0, 0, 0, 0, 0, 0};

    print_array(x, 8);
    printf("\n");

    fft2(8, x, r, i);
    print_array(r, 8);
    printf("\n");
    print_array(i, 8);
    printf("\n");

    return 0;

}

