def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    treatments = list(map(int, data[2:2+Q]))
    
    empty_holes = set()
    
    for T in treatments:
        if T in empty_holes:
            empty_holes.remove(T)
        else:
            empty_holes.add(T)
    
    final_teeth = N - len(empty_holes)
    print(final_teeth)

if __name__ == "__main__":
    main()