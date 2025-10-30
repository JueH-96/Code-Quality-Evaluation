# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = []
    C = []
    for i in range(N):
        A.append(int(data[2*i+1]))
        C.append(int(data[2*i+2]))
    
    # Create a dictionary to map color to list of deliciousness
    color_to_A = {}
    for i in range(N):
        if C[i] in color_to_A:
            color_to_A[C[i]].append(A[i])
        else:
            color_to_A[C[i]] = [A[i]]
    
    # For each color, find the minimum deliciousness
    min_deliciousness = []
    for color in color_to_A:
        min_deliciousness.append(min(color_to_A[color]))
    
    # Find the maximum of these minimums
    result = max(min_deliciousness)
    print(result)

if __name__ == "__main__":
    main()