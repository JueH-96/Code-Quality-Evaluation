import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    votes = list(map(int, data[2:2+m]))
    
    count = [0] * (n + 1)
    max_vote = 0
    best_candidate = 0
    
    out_lines = []
    for c in votes:
        count[c] += 1
        if count[c] > max_vote:
            max_vote = count[c]
            best_candidate = c
        elif count[c] == max_vote:
            if c < best_candidate:
                best_candidate = c
        out_lines.append(str(best_candidate))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()