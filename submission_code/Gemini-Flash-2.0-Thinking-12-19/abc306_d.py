def solve():
    n = int(input())
    courses = []
    for _ in range(n):
        x_i, y_i = map(int, input().split())
        courses.append({'type': x_i, 'tastiness': y_i})
    
    dp = {} # dp[(course_index, stomach_state)] = max_tastiness
    
    def get_dp_value(course_index, stomach_state):
        if course_index == n:
            return 0
        if (course_index, stomach_state) in dp:
            return dp[(course_index, stomach_state)]
        
        current_course = courses[course_index]
        course_type = current_course['type']
        course_tastiness = current_course['tastiness']
        
        # Option 1: Skip the course
        skip_tastiness = get_dp_value(course_index + 1, stomach_state)
        
        # Option 2: Eat the course (if possible)
        eat_tastiness = -float('inf')
        if stomach_state == 0: # Healthy stomach
            next_state_after_eat = course_type # 0 if antidotal, 1 if poisonous
            eat_tastiness = course_tastiness + get_dp_value(course_index + 1, next_state_after_eat)
        elif stomach_state == 1: # Upset stomach
            if course_type == 0: # Antidotal
                next_state_after_eat = 0 # Becomes healthy
                eat_tastiness = course_tastiness + get_dp_value(course_index + 1, next_state_after_eat)
            else: # Poisonous, cannot eat, will die. Only option is to skip.
                eat_tastiness = -float('inf') # Effectively disable eating
                
        result = max(skip_tastiness, eat_tastiness)
        dp[(course_index, stomach_state)] = result
        return result
        
    max_total_tastiness = get_dp_value(0, 0) # Start from course 0 (index 0) with healthy stomach (state 0)
    print(max(0, max_total_tastiness))

if __name__ == '__main__':
    solve()