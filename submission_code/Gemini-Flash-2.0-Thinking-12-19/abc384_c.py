def solve():
    scores = list(map(int, input().split()))
    problems = "ABCDE"
    participants = []
    for i in range(1, 32):
        current_name = ""
        current_score = 0
        for j in range(5):
            if (i >> j) & 1:
                current_name += problems[j]
                current_score += scores[j]
        participants.append((-current_score, current_name))
    
    participants.sort()
    
    for score, name in participants:
        print(name)

if __name__ == '__main__':
    solve()