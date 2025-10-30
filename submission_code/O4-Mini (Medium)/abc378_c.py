import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    
    last_occurrence = {}
    B = []
    
    for i, a in enumerate(A, start=1):
        # Get the last occurrence index or -1 if not seen before
        B.append(str(last_occurrence.get(a, -1)))
        # Update the last occurrence of a
        last_occurrence[a] = i
    
    # Output the result
    print(" ".join(B))

if __name__ == "__main__":
    main()