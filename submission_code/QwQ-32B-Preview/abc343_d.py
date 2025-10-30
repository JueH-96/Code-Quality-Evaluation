def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = int(data[1])
    A_B = list(map(int, data[2:]))
    
    scores = [0] * N
    frequency = {0: N}
    unique = {0}
    
    for i in range(T):
        A = A_B[2*i] - 1  # Convert to 0-based index
        B = A_B[2*i + 1]
        
        old_score = scores[A]
        new_score = old_score + B
        
        # Update frequency
        frequency[old_score] -= 1
        if frequency[old_score] == 0:
            del frequency[old_score]
            unique.remove(old_score)
        if new_score in frequency:
            frequency[new_score] += 1
        else:
            frequency[new_score] = 1
            unique.add(new_score)
        
        scores[A] = new_score
        print(len(unique))

if __name__ == "__main__":
    main()