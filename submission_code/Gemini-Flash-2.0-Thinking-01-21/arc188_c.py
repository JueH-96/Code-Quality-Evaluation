import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    testimonies = []
    for _ in range(m):
        testimonies.append(list(map(int, sys.stdin.readline().split())))
    
    for i in range(1 << n):
        confusion_status = []
        confusion_str = ""
        for j in range(n):
            if (i >> j) & 1:
                confusion_status.append(1)
                confusion_str += "1"
            else:
                confusion_status.append(0)
                confusion_str += "0"
        
        found_valid_assignment = False
        for j in range(1 << n):
            honesty_status = []
            for k in range(n):
                if (j >> k) & 1:
                    honesty_status.append(1) # 1 for honest, 0 for liar
                else:
                    honesty_status.append(0)
            
            is_assignment_valid = True
            for testimony in testimonies:
                speaker_index = testimony[0] - 1
                target_index = testimony[1] - 1
                testimony_type = testimony[2]
                speaker_honesty = honesty_status[speaker_index]
                speaker_confused = confusion_status[speaker_index]
                target_honesty = honesty_status[target_index]
                
                statement_is_honest = (testimony_type == 0)
                
                teller_is_honest = (speaker_honesty == 1)
                teller_is_confused = (speaker_confused == 1)
                
                truth_value = False
                if not teller_is_confused:
                    if teller_is_honest:
                        truth_value = statement_is_honest == (target_honesty == 1)
                    else:
                        truth_value = statement_is_honest != (target_honesty == 1)
                else:
                    if teller_is_honest:
                        truth_value = statement_is_honest != (target_honesty == 1)
                    else:
                        truth_value = statement_is_honest == (target_honesty == 1)
                        
                if not truth_value:
                    is_assignment_valid = False
                    break
                    
            if is_assignment_valid:
                found_valid_assignment = True
                break
                
        if found_valid_assignment:
            print(confusion_str)
            return
            
    print("-1")

if __name__ == '__main__':
    solve()