def main():
    import sys
    S = sys.stdin.readline().strip()
    segments = S.split("|")
    
    # Filter out empty segments and count the number of '-' in each non-empty segment
    A = [len(segment) for segment in segments if segment]
    
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()