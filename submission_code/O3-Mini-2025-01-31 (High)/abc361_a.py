def main():
    # Read integers N (length of sequence), K (position to insert after), X (value to insert)
    N, K, X = map(int, input().split())
    # Read the integer sequence A of length N
    A = list(map(int, input().split()))
    
    # Insert the integer X immediately after the K-th element (1-based index)
    # In Python, list indices start at 0, so the K-th element is at index K-1
    A.insert(K, X)
    
    # Print the resulting sequence with spaces in between
    print(" ".join(map(str, A)))

# Call the main function
if __name__ == '__main__':
    main()