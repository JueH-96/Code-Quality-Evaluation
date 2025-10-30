def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    candidates = set()
    
    # x = 2B - A
    candidates.add(2*B - A)
    
    # x = 2A - B
    candidates.add(2*A - B)
    
    # x = (A + B) / 2 if (A + B) is even
    if (A + B) % 2 == 0:
        candidates.add((A + B) // 2)
    
    print(len(candidates))

# Do not remove the below line
main()