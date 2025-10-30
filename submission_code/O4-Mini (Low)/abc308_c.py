import sys
from functools import cmp_to_key

def main():
    input = sys.stdin.readline
    N = int(input())
    
    people = []
    for idx in range(1, N+1):
        A, B = map(int, input().split())
        people.append((idx, A, B))
    
    def compare(x, y):
        """
        Compare two people x and y.
        x = (idx_x, A_x, B_x)
        y = (idx_y, A_y, B_y)
        We want descending order by A/(A+B), i.e. x before y if
        A_x/(A_x+B_x) > A_y/(A_y+B_y).
        Cross-multiply to avoid floating issues:
        A_x * (A_y+B_y) > A_y * (A_x+B_x)
        """
        idx_x, A_x, B_x = x
        idx_y, A_y, B_y = y
        
        left = A_x * (A_y + B_y)
        right = A_y * (A_x + B_x)
        
        if left > right:
            return -1  # x should come before y
        elif left < right:
            return 1   # y should come before x
        else:
            # tie: ascending by index
            return idx_x - idx_y
    
    people.sort(key=cmp_to_key(compare))
    
    # Extract and print indices in order
    result = [str(p[0]) for p in people]
    print(" ".join(result))

if __name__ == "__main__":
    main()