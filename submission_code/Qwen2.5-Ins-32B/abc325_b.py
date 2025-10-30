import sys

def max_participants(N, WX):
    max_participants = 0
    for start_time in range(24):
        participants = 0
        for W, X in WX:
            meeting_start = (start_time + X) % 24
            if 9 <= meeting_start <= 17:
                participants += W
        max_participants = max(max_participants, participants)
    return max_participants

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()
    index = 0
    N = int(data[index])
    index += 1
    WX = []
    for _ in range(N):
        W = int(data[index])
        index += 1
        X = int(data[index])
        index += 1
        WX.append((W, X))
    print(max_participants(N, WX))