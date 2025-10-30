def main():
    import sys
    S = sys.stdin.readline().rstrip('
')
    
    # Split by '|' to extract blocks of '-' (ignoring the first and last empty segments)
    blocks = S.split('|')[1:-1]
    
    # Each block's length corresponds to an element in A
    A = [len(block) for block in blocks]
    
    # Print the sequence
    print(' '.join(map(str, A)))

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()