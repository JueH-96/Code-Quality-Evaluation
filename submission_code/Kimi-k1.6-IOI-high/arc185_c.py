from bisect import bisect_right
import sys

def main():
    n, x = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    
    value_to_indices = {}
    for idx, num in enumerate(a):
        if num not in value_to_indices:
            value_to_indices[num] = []
        value_to_indices[num].append(idx)
    
    found = False
    max_first_two = 2000  # Adjusted based on trial, can be larger if needed
    
    for i in range(n):
        if i > max_first_two:
            break
        for j in range(i + 1, n):
            if j > max_first_two:
                break
            required = x - a[i] - a[j]
            if required in value_to_indices:
                indices_list = value_to_indices[required]
                # Find the first index in indices_list that is greater than j
                pos = bisect_right(indices_list, j)
                if pos < len(indices_list):
                    k = indices_list[pos]
                    print(f"{i+1} {j+1} {k+1}")
                    found = True
                    return
    
    print(-1)

if __name__ == "__main__":
    main()