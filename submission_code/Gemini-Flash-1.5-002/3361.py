class Solution:
    def findLatestTime(self, s: str) -> str:
        hours, minutes = s.split(":")
        hours_list = list(hours)
        minutes_list = list(minutes)

        def replace_question_marks(lst, upper_bound):
            for i in range(len(lst)):
                if lst[i] == '?':
                    if i == 0 and upper_bound >= 10:
                        lst[i] = '1'
                    elif i == 0 and upper_bound < 10:
                        lst[i] = '0' if upper_bound == 0 else str(upper_bound -1)
                    elif i == 1:
                        lst[i] = str(upper_bound % 10)
            return "".join(lst)

        
        hours_int = int("".join(hours_list)) if "?" not in hours else -1
        minutes_int = int("".join(minutes_list)) if "?" not in minutes else -1

        if hours_int == -1:
            hours = replace_question_marks(hours_list,11)
        elif hours_int > 11:
            hours = "11"
        
        if minutes_int == -1:
            minutes = replace_question_marks(minutes_list,59)
        elif minutes_int > 59:
            minutes = "59"

        if int(hours) > 11:
            hours = "11"
            minutes = "59"
        elif int(minutes) > 59:
            minutes = "59"

        return hours + ":" + minutes