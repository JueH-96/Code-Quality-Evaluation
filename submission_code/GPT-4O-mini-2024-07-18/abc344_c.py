def can_form_sum(A, B, C, X):
    possible_sums = set()
    
    # Generate all possible sums of one element from A and one from B
    for a in A:
        for b in B:
            possible_sums.add(a + b)
    
    # Now check if we can form X_i using one element from C
    results = []
    for x in X:
        found = False
        for c in C:
            if (x - c) in possible_sums:
                found = True
                break
        results.append("Yes" if found else "No")
    
    return results

import sys
input = sys.stdin.read

def main():
    data = input().splitlines()
    index = 0
    
    N = int(data[index])
    index += 1
    A = list(map(int, data[index].split()))
    index += 1
    
    M = int(data[index])
    index += 1
    B = list(map(int, data[index].split()))
    index += 1
    
    L = int(data[index])
    index += 1
    C = list(map(int, data[index].split()))
    index += 1
    
    Q = int(data[index])
    index += 1
    X = list(map(int, data[index].split()))
    
    results = can_form_sum(A, B, C, X)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()