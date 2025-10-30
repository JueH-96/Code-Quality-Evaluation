def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    
    candidates = set()
    
    # 1) x = 2B - A
    candidates.add(2*B - A)
    # 2) x = 2A - B
    candidates.add(2*A - B)
    # 3) x = (A + B) / 2 if A+B is even
    if (A + B) % 2 == 0:
        candidates.add((A + B) // 2)
    
    print(len(candidates))

# Do not remove this call
main()