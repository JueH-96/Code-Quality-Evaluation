def solve():
    scores = list(map(int, input().split()))
    problem_scores = {'A': scores[0], 'B': scores[1], 'C': scores[2], 'D': scores[3], 'E': scores[4]}
    problems_list = ['A', 'B', 'C', 'D', 'E']
    participants_data = []
    for i in range(1, 32):
        binary_i = format(i, '05b')
        name = ""
        score = 0
        for j in range(5):
            if binary_i[j] == '1':
                name += problems_list[j]
                score += problem_scores[problems_list[j]]
        participants_data.append({'name': name, 'score': score})
    
    def compare_participants(p1, p2):
        if p1['score'] != p2['score']:
            return p2['score'] - p1['score']
        else:
            if p1['name'] < p2['name']:
                return -1
            elif p1['name'] > p2['name']:
                return 1
            else:
                return 0
                
    from functools import cmp_to_key
    participants_data.sort(key=cmp_to_key(compare_participants))
    
    for participant in participants_data:
        print(participant['name'])

if __name__ == '__main__':
    solve()