import bisect
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    n = int(input[ptr])
    ptr += 1
    m = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr + n]))
    ptr += n
    
    B = list(map(int, input[ptr:ptr + m]))
    ptr += m
    
    people = [(A[i], i + 1) for i in range(n)]
    people_sorted = sorted(people, key=lambda x: (x[0], x[1]))
    sorted_values = [x[0] for x in people_sorted]
    
    min_indices = [0] * n
    min_indices[0] = people_sorted[0][1]
    for i in range(1, n):
        min_indices[i] = min(min_indices[i - 1], people_sorted[i][1])
    
    for b in B:
        k = bisect.bisect_right(sorted_values, b)
        if k == 0:
            print(-1)
        else:
            print(min_indices[k - 1])

if __name__ == "__main__":
    main()