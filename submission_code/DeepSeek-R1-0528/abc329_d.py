import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    votes = list(map(int, data[2:2+m]))
    
    count = [0] * (n+1)
    current_max = 0
    current_winner = 0
    results = []
    
    for a in votes:
        count[a] += 1
        if count[a] > current_max:
            current_max = count[a]
            current_winner = a
        elif count[a] == current_max:
            if a < current_winner:
                current_winner = a
        results.append(current_winner)
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()