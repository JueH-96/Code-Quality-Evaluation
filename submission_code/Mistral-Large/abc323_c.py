import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0

    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1

    A = [int(data[index + i]) for i in range(M)]
    index += M

    S = [data[index + i] for i in range(N)]
    index += N

    # Calculate the current scores
    current_scores = [0] * N
    for i in range(N):
        for j in range(M):
            if S[i][j] == 'o':
                current_scores[i] += A[j]
        current_scores[i] += (i + 1)

    # Calculate the answers
    answers = []
    for i in range(N):
        max_other_score = max(current_scores[:i] + current_scores[i+1:])
        unsolved_problems = [A[j] for j in range(M) if S[i][j] == 'x']
        unsolved_problems.sort(reverse=True)

        additional_score_needed = max_other_score - current_scores[i] + 1
        count = 0
        while additional_score_needed > 0:
            additional_score_needed -= unsolved_problems[count]
            count += 1

        answers.append(count)

    # Output the results
    for answer in answers:
        print(answer)

if __name__ == "__main__":
    solve()