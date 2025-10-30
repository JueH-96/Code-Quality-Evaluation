import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    T = int(data[idx])
    idx += 1
    
    scores = [0] * (N + 1)
    frequencies = {}
    frequencies[0] = N
    
    for _ in range(T):
        A = int(data[idx])
        idx += 1
        B = int(data[idx])
        idx += 1
        
        old_score = scores[A]
        # Update old_score frequency
        cnt = frequencies[old_score]
        cnt -= 1
        if cnt == 0:
            del frequencies[old_score]
        else:
            frequencies[old_score] = cnt
        
        # Update the score
        new_score = old_score + B
        scores[A] = new_score
        
        # Update new_score frequency
        frequencies[new_score] = frequencies.get(new_score, 0) + 1
        
        # Output the number of distinct scores
        print(len(frequencies))

if __name__ == '__main__':
    main()