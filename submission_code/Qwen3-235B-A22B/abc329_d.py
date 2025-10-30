import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    m = int(input[1])
    votes = list(map(int, input[2:2+m]))
    
    counts = [0] * (n + 1)
    current_winner = 0
    max_count = 0
    output = []
    
    for a in votes:
        counts[a] += 1
        if counts[a] > max_count:
            current_winner = a
            max_count = counts[a]
        elif counts[a] == max_count:
            if a < current_winner:
                current_winner = a
        output.append(str(current_winner))
    
    print('
'.join(output))

if __name__ == '__main__':
    main()