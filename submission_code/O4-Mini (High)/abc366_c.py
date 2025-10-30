import sys
import threading

def main():
    import sys
    input = sys.stdin.readline
    
    Q = int(input())
    # Dictionary to store counts of each integer in the bag
    counts = {}
    # Number of distinct integers currently in the bag
    distinct = 0
    
    out = []
    for _ in range(Q):
        query = input().split()
        t = int(query[0])
        
        if t == 1:
            # Insert a ball labeled x
            x = int(query[1])
            # If x is not present or its count was zero, we are introducing a new distinct value
            if counts.get(x, 0) == 0:
                distinct += 1
            counts[x] = counts.get(x, 0) + 1
        
        elif t == 2:
            # Remove a ball labeled x
            x = int(query[1])
            # We know counts[x] >= 1
            counts[x] -= 1
            # If it drops to zero, one distinct value disappears
            if counts[x] == 0:
                distinct -= 1
        
        else:  # t == 3
            # Report the number of distinct integers
            out.append(str(distinct))
    
    # Print all answers for type-3 queries
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()