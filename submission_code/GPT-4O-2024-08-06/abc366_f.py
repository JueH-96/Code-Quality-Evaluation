import sys
import itertools

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    
    functions = []
    index = 2
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        functions.append((A, B))
        index += 2
    
    max_value = float('-inf')
    
    # Generate all permutations of length K from the range 0 to N-1
    for perm in itertools.permutations(range(N), K):
        value = 1
        for idx in perm:
            A, B = functions[idx]
            value = A * value + B
        max_value = max(max_value, value)
    
    print(max_value)

if __name__ == "__main__":
    main()