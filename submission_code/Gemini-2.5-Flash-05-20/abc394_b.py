import sys

def solve():
    N = int(sys.stdin.readline())
    
    strings = []
    for _ in range(N):
        strings.append(sys.stdin.readline().strip())
        
    # Sort strings by their length in ascending order
    # The problem guarantees that lengths are all distinct,
    # so the sort order is unique.
    strings.sort(key=len)
    
    # Concatenate the sorted strings
    result = "".join(strings)
    
    print(result)

if __name__ == '__main__':
    solve()