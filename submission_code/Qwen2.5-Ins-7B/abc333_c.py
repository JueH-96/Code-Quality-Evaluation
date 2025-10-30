# YOUR CODE HERE
def generate_repunit(n):
    return int('1' * n)

def find_nth_sum(n):
    count = 0
    i = 1
    while True:
        repunit = generate_repunit(i)
        for j in range(i, -1, -1):
            repunit_j = generate_repunit(j)
            for k in range(j, -1, -1):
                repunit_k = generate_repunit(k)
                if repunit + repunit_j + repunit_k > 0 and count == n:
                    return repunit + repunit_j + repunit_k
                count += 1
        i += 1

N = int(input())
print(find_nth_sum(N))