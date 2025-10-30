def solve():
    N = int(input())
    P = list(map(int, input().split()))
    Q = list(map(int, input().split()))
    
    # person_to_bib[i] gives the bib number of person i
    person_to_bib = {}
    for i in range(N):
        person_to_bib[i+1] = Q[i]
    
    # bib_to_person[i] gives the person number wearing bib i
    bib_to_person = {}
    for i in range(N):
        bib_to_person[Q[i]] = i + 1
    
    answers = []
    for i in range(1, N+1):
        person = bib_to_person[i]  # person wearing bib i
        person_staring_at = P[person-1]  # who that person is staring at
        bib_of_staring_at = person_to_bib[person_staring_at]  # bib of the person being stared at
        answers.append(str(bib_of_staring_at))
    
    print(" ".join(answers))

solve()