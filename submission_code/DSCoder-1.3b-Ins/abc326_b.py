# YOUR CODE HERE

def find_smallest_326_like_number(N):
    N = str(N)
    for i in range(int(N), 1000):
        str_i = str(i)
        if str_i[0] != '0' and str_i[1] != '0' and str_i[2] != '0' and str_i[0]*2 == str_i[1] and str_i[1]*2 == str_i[2] and str_i[0]*3 == str_i[2]:
            return i
    return -1

N = int(input())
print(find_smallest_326_like_number(N))