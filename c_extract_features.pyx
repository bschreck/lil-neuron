# cython: profile=True
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.math cimport pow
from libc.string cimport strcmp
from libc.string cimport strlen
from libc.stdio cimport sscanf
from cpython.string cimport PyString_AsString

DTYPE_INT = np.int
DTYPE_FLOAT = np.float
ctypedef np.int_t DTYPE_t
ctypedef np.float_t FLOAT_t

# cdef packed struct Counts:
    # DTYPE_t x,y


@cython.boundscheck(True)
def extract_raw_words(np_word_arr):
    return np_word_arr

# @cython.boundscheck(False)
# def expanding_total_and_percentage_ohe(list_str):
    # cdef unsigned int i, j, k

    # cdef unsigned int n = len(list_str)
    # #list of pointers to int arrays
    # cdef int **values = <int **>malloc(n * sizeof(int *))
    # cdef int *int_array_lengths = <int *>malloc(n * sizeof(int))
    # cdef np.ndarray[DTYPE_t, ndim=1] new_series = np.zeros(n, dtype=DTYPE_INT)

    # py_list_to_int_arrays(list_str, values,int_array_lengths)



    # cdef int lenUnique = unique_ohe(n, values, int_array_lengths)

    # cdef np.ndarray[DTYPE_t, ndim=1] counts = np.zeros(lenUnique,dtype=DTYPE_INT)
    # cdef np.ndarray[DTYPE_t, ndim=2] totals = np.zeros((lenUnique,n), dtype=DTYPE_INT)
    # cdef np.ndarray[FLOAT_t, ndim=2] percentages = np.zeros((lenUnique,n), dtype=DTYPE_FLOAT)
    # cdef unsigned int found = 0

    # #map actual values to indices from 0->lenUnique
    # cdef unsigned int current_mapping = 0
    # cdef np.ndarray[DTYPE_t, ndim=1] mapping = np.zeros(lenUnique,dtype=DTYPE_INT)
    # for i in xrange(n):
        # for j in xrange(int_array_lengths[i]):
            # for k in xrange(current_mapping):
                # if mapping[k] == values[i][j]:
                    # found = 1
                    # break
            # if found == 0:
                # mapping[current_mapping] = values[i][j]
                # current_mapping += 1
            # found = 0


    # for i in xrange(n):
        # for j in xrange(int_array_lengths[i]):
            # for k in xrange(lenUnique):
                # if mapping[k] == values[i][j]:
                    # counts[k] += 1
                # totals[k,i] = counts[k]
                # percentages[k,i] = np.float(counts[k]) / (i+1)
    # return totals, percentages, mapping

# @cython.boundscheck(False)
# def expanding_total_and_percentage(np.ndarray[DTYPE_t, ndim=1] values):
    # assert values.dtype == DTYPE_INT
    # cdef unsigned int lenUnique = len(np.unique(values))
    # cdef unsigned int lenValues = len(values)
    # cdef np.ndarray[DTYPE_t, ndim=1] counts = np.zeros(lenUnique,dtype=DTYPE_INT)
    # cdef np.ndarray[DTYPE_t, ndim=2] totals = np.zeros((lenUnique,lenValues), dtype=DTYPE_INT)
    # cdef np.ndarray[FLOAT_t, ndim=2] percentages = np.zeros((lenUnique,lenValues), dtype=DTYPE_FLOAT)
    # cdef unsigned int i
    # cdef unsigned int j
    # cdef unsigned int found = 0

    # #map actual values to indices from 0->lenUnique
    # cdef unsigned int current_mapping = 0
    # cdef np.ndarray[DTYPE_t, ndim=1] mapping = np.zeros(lenUnique,dtype=DTYPE_INT)
    # for i in range(len(values)):
        # for j in range(current_mapping):
            # if mapping[j] == values[i]:
                # found = 1
                # break
        # if found == 0:
            # mapping[current_mapping] = values[i]
            # current_mapping += 1
        # found = 0


    # for i in range(len(values)):
        # for j in range(lenUnique):
            # if mapping[j] == values[i]:
                # counts[j] += 1
            # totals[j,i] = counts[j]
            # percentages[j,i] = np.float(counts[j]) / (i+1)
    # return totals, percentages, mapping


# @cython.boundscheck(False)
# def expanding_mode(np.ndarray[DTYPE_t, ndim=1] values):
    # assert values.dtype == DTYPE_INT
    # cdef int lenUnique = len(np.unique(values))

    # cdef unsigned int i
    # cdef np.ndarray[Counts] sorted_counts = np.zeros(lenUnique,dtype=np.dtype([('x',DTYPE_INT),('y',DTYPE_INT)]))
    # for i in range(len(sorted_counts)):
        # sorted_counts[i].x = -1
        # sorted_counts[i].y = -1

    # i = 0
    # for i in range(len(values)):
        # sorted_counts = updateSortedCounts(values[i], sorted_counts)
        # values[i] = sorted_counts[lenUnique-1].x
    # return values

# @cython.boundscheck(False)
# def expanding_mean(np.ndarray[FLOAT_t, ndim=1] values, DTYPE_t absolute_value):
    # assert values.dtype == DTYPE_FLOAT

    # cdef unsigned int n = len(values)
    # cdef np.ndarray[FLOAT_t, ndim=1] mean = np.zeros(n, dtype=DTYPE_FLOAT)
    # cdef unsigned int i
    # mean[0] = values[0]
    # if absolute_value and values[0] < 0:
        # mean[0] = -values[0]
    # for i in xrange(1,n):
        # if absolute_value and values[i] < 0:
            # mean[i] = mean[i-1]-values[i]
        # else:
            # mean[i] = mean[i-1]+values[i]
    # for i in range(1,n):
        # mean[i] /= (i+1)
    # return mean

# @cython.boundscheck(False)
# def expanding_var(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT

    # cdef unsigned int n = len(values)
    # cdef np.ndarray[FLOAT_t, ndim=1] mean = expanding_mean(values,0)
    # cdef np.ndarray[FLOAT_t, ndim=1] variance = np.zeros(n, dtype=DTYPE_FLOAT)
    # cdef FLOAT_t intermediate_total = 0
    # cdef unsigned int i,j

    # variance[0] = 0
    # for i in xrange(1,n):
        # intermediate_total = 0
        # for j in xrange(i+1):
            # intermediate_total +=  pow(values[j]-mean[i], 2)
        # variance[i] = intermediate_total/(i+1)
    # return variance


# @cython.boundscheck(False)
# def expanding_diff(np.ndarray[FLOAT_t, ndim=1] values, DTYPE_t absolute_value):
    # assert values.dtype == DTYPE_FLOAT

    # cdef unsigned int n = len(values)
    # cdef np.ndarray[FLOAT_t, ndim=1] diff = np.zeros(n, dtype=DTYPE_FLOAT)
    # cdef unsigned int i
    # cdef FLOAT_t to_check
    # diff[0] = 0
    # for i in xrange(1,n):
        # diff[i] = values[i] - values[i-1]
        # if absolute_value and diff[i] < 0:
            # diff[i] = -diff[i]
    # return diff

# @cython.boundscheck(False)
# def expanding_abs_min_diff_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,0)
    # result = expanding_diff(result,1)
    # result = expanding_min(result,0,1)
    # return result

# @cython.boundscheck(False)
# def expanding_abs_max_diff_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,0)
    # result = expanding_diff(result,1)
    # result = expanding_max(result,0)
    # return result

# @cython.boundscheck(False)
# def expanding_abs_mean_diff_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,0)
    # result = expanding_diff(result,1)
    # result = expanding_mean(result,0)
    # return result

# @cython.boundscheck(False)
# def expanding_max_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,0)
    # result = expanding_max(result,0)
    # return result

# @cython.boundscheck(False)
# def expanding_abs_mean_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,1)
    # result = expanding_mean(result,0)
    # return result

# @cython.boundscheck(False)
# def expanding_abs_max_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,1)
    # result = expanding_max(result,0)
    # return result

# @cython.boundscheck(False)
# def expanding_min_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,0)
    # result = expanding_min(result,0,1)
    # return result

# @cython.boundscheck(False)
# def expanding_abs_min_diff(np.ndarray[FLOAT_t, ndim=1] values):
    # assert values.dtype == DTYPE_FLOAT
    # cdef np.ndarray[FLOAT_t, ndim=1] result = expanding_diff(values,1)
    # result = expanding_min(result,0,1)
    # return result

# @cython.boundscheck(False)
# def expanding_sum(np.ndarray[FLOAT_t, ndim=1] values, DTYPE_t absolute_value):
    # assert values.dtype == DTYPE_FLOAT

    # cdef unsigned int n = len(values)
    # cdef unsigned int i
    # if absolute_value and values[0] < 0:
        # values[0] = -values[0]
    # for i in xrange(1,n):
        # if absolute_value and values[i] < 0:
            # values[i] = -values[i] + values[i-1]
        # else:
            # values[i] += values[i-1]
    # return values

# @cython.boundscheck(False)
# def expanding_max(np.ndarray[FLOAT_t, ndim=1] values, DTYPE_t absolute_value):
    # assert values.dtype == DTYPE_FLOAT

    # cdef unsigned int n = len(values)
    # cdef unsigned int i
    # cdef FLOAT_t to_check
    # if absolute_value and values[0] < 0:
        # values[0] = -values[0]
    # cdef FLOAT_t current_max = values[0]
    # for i in xrange(1,n):
        # to_check = values[i]
        # if absolute_value and values[i] < 0:
            # to_check = -values[i]
        # if to_check > current_max:
            # current_max = to_check
        # else:
            # values[i] = values[i-1]
    # return values

# @cython.boundscheck(False)
# def expanding_min(np.ndarray[FLOAT_t, ndim=1] values, DTYPE_t absolute_value, DTYPE_t ignore_first_value):
    # assert values.dtype == DTYPE_FLOAT

    # cdef unsigned int n = len(values)
    # cdef unsigned int i
    # cdef unsigned int start_index = 0
    # cdef FLOAT_t to_check
    # if ignore_first_value:
        # start_index = 1
    # if absolute_value and values[start_index] < 0:
        # values[start_index] = -values[start_index]
    # cdef FLOAT_t current_min = values[start_index]
    # for i in xrange(start_index+1,n):
        # to_check = values[i]
        # if absolute_value and values[i] < 0:
            # to_check = -values[i]
        # if to_check < current_min:
            # current_min = to_check
        # else:
            # values[i] = values[i-1]
    # return values

# @cython.boundscheck(False)
# def expanding_jitter(np.ndarray[DTYPE_t, ndim=1] values):
    # assert values.dtype == DTYPE_INT
    # cdef unsigned int i
    # cdef unsigned int n = len(values)
    # cdef np.ndarray[FLOAT_t, ndim=1] jitter = np.zeros(n, dtype=DTYPE_FLOAT)
    # for i in range(1, n):
        # jitter[i] = jitter[i-1]
        # if values[i] != values[i-1]:
            # jitter[i] += 1
    # for i in range(1,n):
        # jitter[i] /= (i+1)
    # return jitter

# @cython.boundscheck(False)
# def expanding_stability(np.ndarray[DTYPE_t, ndim=1] values):
    # assert values.dtype == DTYPE_INT
    # cdef int lenUnique = len(np.unique(values))

    # cdef unsigned int i,j
    # cdef unsigned int n = len(values)
    # cdef np.ndarray[FLOAT_t, ndim=1] stability = np.zeros(n, dtype=DTYPE_FLOAT)
    # cdef np.ndarray[Counts] sorted_counts = np.zeros(lenUnique,dtype=np.dtype([('x',DTYPE_INT),('y',DTYPE_INT)]))
    # for i in range(len(sorted_counts)):
        # sorted_counts[i].x = -1
        # sorted_counts[i].y = -1

    # i = 0
    # for i in range(1, len(values)):
        # sorted_counts = updateSortedCounts(values[i], sorted_counts)
        # for j in range(len(sorted_counts)-1):
            # if sorted_counts[j].y > -1:
                # stability[i] += sorted_counts[j].y

    # stability[0] = 1
    # for i in range(1,n):
        # stability[i] /= (i+1)
        # stability[i] = 1-stability[i]
    # return stability

# @cython.boundscheck(False)
# def expanding_mode_ohe(list_str):
    # cdef unsigned int i, j

    # cdef unsigned int n = len(list_str)
    # #list of pointers to int arrays
    # cdef int **values = <int **>malloc(n * sizeof(int *))
    # cdef int *int_array_lengths = <int *>malloc(n * sizeof(int))
    # cdef np.ndarray[DTYPE_t, ndim=1] new_series = np.zeros(n, dtype=DTYPE_INT)

    # py_list_to_int_arrays(list_str, values,int_array_lengths)



    # cdef int lenUnique = unique_ohe(n, values, int_array_lengths)

    # cdef np.ndarray[Counts] sorted_counts = np.zeros(lenUnique,dtype=np.dtype([('x',DTYPE_INT),('y',DTYPE_INT)]))
    # for i in xrange(len(sorted_counts)):
        # sorted_counts[i].x = -1
        # sorted_counts[i].y = -1

    # i = 0
    # for i in xrange(n):
        # for j in xrange(int_array_lengths[i]):
            # updateSortedCounts(values[i][j],sorted_counts)
        # free(values[i])
        # new_series[i] = sorted_counts[lenUnique-1].x
    # free(values)
    # return new_series



# @cython.boundscheck(False)
# def expanding_jitter_ohe(list_str):
    # cdef unsigned int i,j
    # cdef unsigned int n = len(list_str)
    # cdef int **values = <int **>malloc(n * sizeof(int *))
    # cdef int *int_array_lengths = <int *>malloc(n * sizeof(int))
    # cdef np.ndarray[FLOAT_t, ndim=1] jitter = np.zeros(n, dtype=DTYPE_FLOAT)

    # py_list_to_int_arrays(list_str, values,int_array_lengths)
    # for i in range(1, n):
        # jitter[i] = jitter[i-1]
        # if int_array_lengths[i] != int_array_lengths[i-1]:
            # jitter[i] += 1
        # else:
            # for j in xrange(int_array_lengths[i]):
                # if values[i][j] != values[i-1][j]:
                    # jitter[i] += 1
                    # break
        # free(values[i-1])
    # free(values[n-1])
    # free(values)
    # for i in range(1,n):
        # jitter[i] /= (i+1)
    # return jitter

# @cython.boundscheck(False)
# @cython.profile(False)
# cdef inline np.ndarray[Counts] updateSortedCounts(int val, np.ndarray[Counts] sorted_counts):
    # cdef unsigned int valIdx = 0
    # cdef unsigned int found = 0
    # cdef unsigned int temp = 0
    # for valIdx in xrange(len(sorted_counts)):
        # if sorted_counts[valIdx].x == val:
            # sorted_counts[valIdx].y += 1
            # found = 1
            # break
    # if found == 0:
        # sorted_counts[0].x = val
        # sorted_counts[0].y = 1
        # return insertionSortInner(1, sorted_counts)
    # else:
        # return insertionSortInner(valIdx+1, sorted_counts)

# @cython.boundscheck(False)
# @cython.profile(False)
# cdef inline np.ndarray[Counts] insertionSortInner(int startIdx, np.ndarray[Counts] sorted_counts):
    # cdef unsigned int idx
    # for idx in xrange(startIdx,len(sorted_counts)):
        # if sorted_counts[idx].y < sorted_counts[idx -1].y:
            # temp = sorted_counts[idx].y
            # sorted_counts[idx].y = sorted_counts[idx-1].y
            # sorted_counts[idx-1].y = temp

            # temp = sorted_counts[idx].x
            # sorted_counts[idx].x = sorted_counts[idx-1].x
            # sorted_counts[idx-1].x = temp
        # else:
            # break
    # return sorted_counts

# @cython.boundscheck(False)
# @cython.profile(False)
# cdef inline void py_list_to_int_arrays(list_str, int** int_array_list, int* int_array_lengths):
    # #converts Python list of strings,
    # #with each string itself a comma separated list of ints
    # #into a C int** array of ptrs to int* arrays,
    # #with each int* array containing a grouping
    # #of integers from the original indexed comma separated list

    # cdef int n = len(list_str)
    # cdef unsigned int i,j,k,l = 0
    # cdef unsigned int found_comma = 0
    # cdef char **ret = <char **>malloc(n * sizeof(char *))

    # #container to hold string versions of each number, assume
    # #numbers don't get bigger than 1,000,000,000:
    # cdef char tmp_str[10]
    # cdef char tmp
    # for i in xrange(n):
        # ret[i] = PyString_AsString(list_str[i])

        # #assume there's at least 1 int, need to make sure this is true in the python
        # int_array_lengths[i] = 1
        # for j in xrange(strlen(ret[i])):
            # if ret[i][j] == ',':
                # int_array_lengths[i] += 1
                # found_comma = 1
        # #if last element is a comma, then we added one too many
        # if ret[i][strlen(ret[i])-1] == ',':
            # int_array_lengths[i] -= 1
        # int_array_list[i] = <int *>malloc(int_array_lengths[i] * sizeof(int))

        # #for each comma separated number in the char array
        # #j loops through numbers

        # #l indexes start of each number in ret[i]
        # l=0
        # for j in xrange(int_array_lengths[i]):
            # #k loops through chars
            # #loop through the char array until next comma
            # k = 0
            # tmp = '0'
            # while tmp != ',' and tmp != '\0':
                # #add current char to tmp_str
                # tmp_str[k] = ret[i][l+k]
                # k += 1
                # #add next char to tmp for next comma check
                # tmp = ret[i][l+k]
            # #null terminate the string so we can convert
            # tmp_str[k] = '\0'
            # l += k + 1
            # #tmp was comma so increment to next char
            # #we found a full int, so cast to int and place in int_array
            # #j is the offset from from int array int_array_list[i]
            # sscanf(tmp_str, "%d", &int_array_list[i][j])
    # free(ret)

# @cython.boundscheck(False)
# @cython.profile(False)
# cdef inline int unique_ohe(int n, int** int_array_list, int* int_array_lengths):
    # cdef unsigned int maxUnique = 0
    # cdef unsigned int numUniques = 0
    # cdef unsigned int i,j,foundDup

    # for i in xrange(n):
        # maxUnique += int_array_lengths[i]

    # cdef np.ndarray[DTYPE_t] uniques = np.zeros(maxUnique,dtype=DTYPE_INT)
    # for i in xrange(maxUnique):
        # uniques[i] = -1
    # for i in xrange(n):
        # for j in xrange(int_array_lengths[i]):
            # found_dup = 0
            # for k in xrange(numUniques):
                # if int_array_list[i][j] == uniques[k]:
                    # found_dup = 1
                    # break
            # if found_dup == 0:
                # uniques[numUniques] = int_array_list[i][j]
                # numUniques += 1
    # return numUniques

# #@cython.boundscheck(False)
# #@cython.profile(False)
# #cdef inline int length_ohe(int n, int** int_array_list, int* int_array_lengths):
    # #cdef unsigned int length = 0
    # #cdef unsigned int i,j

    # #for i in xrange(n):
        # #length += int_array_lengths[i]
    # #return length
