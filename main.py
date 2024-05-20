import csv
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from pytz import timezone
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

#move this over
class ClassData:
    def __init__(self, course_name, credits, difficulty, current_grade, study_hours):
        self.course_name = course_name
        self.credits = credits
        self.difficulty = difficulty
        self.current_grade = current_grade
        self.study_hours = study_hours
        self.recommended_hours = 0  # This will store the recommended study hours

# forst 3 lines move over
class Chatbot:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.classes = []
        self.schedule = {day: [] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}
        self.pst = timezone('US/Pacific')

    def collect_class_data(self):
        while True:
            print("Please enter the course data or type 'done' to finish.")
            course_name = input("What's the name of your course? ")
            if course_name.lower() == 'done':
                break
            credits = float(input("How many credits is the course? "))
            difficulty = int(input("On a scale of 1 to 10, how difficult is the course? "))
            current_grade = float(input("What's your current grade in the course ((e.g. 0-100)if applicable)? "))
            study_hours = float(input("How many hours do you study for this course per week? "))
            self.classes.append(ClassData(course_name, credits, difficulty, current_grade, study_hours))
            print("Course added. You can add another course or type 'done'.")
            
            # move to the other add info path in the API
            # Append class data to CSV file
            with open('class_data.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([course_name, credits, difficulty, current_grade, study_hours])
                
            # Update the model with new data
            self.update_model()

    def update_model(self):
        X = [[data.credits, data.difficulty, data.current_grade] for data in self.classes]
        y = [data.study_hours for data in self.classes]
        self.scaler.fit(X)  # Fit the scaler with the training data
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True

    def get_study_hours_recommendations(self):
        if not self.is_fitted:
            print("No data to make recommendations. Please add some class data first!")
            return
        for data in self.classes: # 65 - 78 move
            input_features = self.scaler.transform([[data.credits, data.difficulty, data.current_grade]])
            data.recommended_hours = self.model.predict(input_features)[0]

            grade_gap = 100 - data.current_grade
            difficulty_factor = data.difficulty / 10
            additional_hours = grade_gap * 0.5 * difficulty_factor
            recommended_hours = max(data.recommended_hours + additional_hours, data.study_hours + 1)

            # Round off the recommended hours by half-hour intervals
            rounded_hours = round(recommended_hours * 2) / 2

            # Update the recommended hours
            data.recommended_hours = rounded_hours

            print(f"Recommended study hours for {data.course_name}: {data.recommended_hours:.2f} hours per week.")

    def collect_schedule(self):
        print("Please enter your busy hours for each day of the week in Pacific Standard Time (PST).")
        print("Use 12-hour format (e.g.,'9:00AM-10:15AM, 1:30PM-3:00PM'). If you have no commitments, enter 'none'.")
        for day in self.schedule:
            busy_times = input(f"{day}: ")
            if busy_times.lower() == 'none':
                self.schedule[day] = []
            else:
                busy_hours = []
                for hour in busy_times.split(', '):
                    start_str, end_str = re.split(r'[-,]', hour.strip())
                    start_time = datetime.strptime(start_str.strip(), '%I:%M%p').replace(tzinfo=self.pst)
                    end_time = datetime.strptime(end_str.strip(), '%I:%M%p').replace(tzinfo=self.pst)
                    busy_hours.append((start_time.hour, start_time.minute, end_time.hour, end_time.minute))
                self.schedule[day] = busy_hours

    def allocate_study_hours(self): #move to website
        total_hours = {day: [] for day in self.schedule}  # Prepare list of hours per course per day
        total_available_hours = {
            day: 24 - sum((end_hour - start_hour) * 60 + (end_minute - start_minute) for start_hour, start_minute, end_hour, end_minute in self.schedule[day]) / 60 if self.schedule[day] else 24
            for day in self.schedule
        }

        total_week_hours = sum(total_available_hours.values())
        for day in total_hours:
            for course in self.classes:
                daily_hours = (course.recommended_hours / total_week_hours) * total_available_hours[day]
                total_hours[day].append((course.course_name, daily_hours))

        return total_hours


    def suggest_study_hours(self):
        available_study_slots = {}
        for day, courses in self.allocate_study_hours().items():
            print(f"Suggested study hours on {day}:")
            for course, hours in courses:
                # Round off the suggested study hours by half-hour intervals
                rounded_hours = round(hours * 2) / 2
                print(f"  {course}: {rounded_hours:.2f} hours")
            
            if not self.schedule[day]:  # Check if there are no commitments for the day
                available_study_slots[day] = [(0, 0, 23, 59)]  # Set end hour to 23 and end minute to 59
            else:
                available_slots = self.find_available_slots(day)
                if available_slots:
                    available_study_slots[day] = available_slots
                else:
                    available_study_slots[day] = []

        # Ask for relaxation technique after displaying suggestions for the whole week
        print("\nWould you like to take a break and do a relaxation technique during any available study slot this week? (yes/no)")
        choice = input().lower()
        if choice == 'yes':
            for day, slots in available_study_slots.items():
                if slots:
                    print(f"\nAvailable study slots on {day}:")
                    for i, (start_hour, start_minute, end_hour, end_minute) in enumerate(slots, start=1):
                        start_time = datetime(2024, 1, 1, start_hour, start_minute, tzinfo=self.pst)
                        end_time = datetime(2024, 1, 1, end_hour, end_minute, tzinfo=self.pst)
                        print(f"{i}. {start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}")

                    print(f"Would you like to take a break on {day}? (yes/no)")
                    break_choice = input().lower()
                    if break_choice == 'yes':
                        print("\nWould you like to try a relaxation technique?")
                        print("1. Breathing Exercise")
                        print("2. Meditation Session")
                        technique_choice = input("Enter the number of your choice (1 or 2), or type 'exit' to return to the main menu: ")
                        if technique_choice in ('1', '2'):
                            self.offer_relaxation_techniques(int(technique_choice), slots)

    def offer_relaxation_techniques(self, choice, slots):
        print("\nLet's practice a relaxation technique:")
        if choice == 1:
            print("Breathing Exercise:")
            print("1. Find a comfortable seated position.")
            print("2. Close your eyes and take a deep breath in through your nose, counting to four.")
            print("3. Hold your breath for a count of four.")
            print("4. Exhale slowly through your mouth, counting to four.")
            print("5. Repeat this cycle for several minutes.")
        elif choice == 2:
            print("Meditation Session:")
            print("1. Find a quiet and comfortable place to sit.")
            print("2. Close your eyes and take slow, deep breaths.")
            print("3. Focus your attention on your breath or a calming object.")
            print("4. Allow your mind to relax and let go of any thoughts.")
            print("5. Continue meditating for 5 to 10 minutes.")

    def find_available_slots(self, day):
        available_slots = []
        busy_slots = self.schedule[day]
        for i in range(len(busy_slots) - 1):
            start_hour, start_minute, end_hour, end_minute = busy_slots[i]
            next_start_hour, next_start_minute, _, _ = busy_slots[i + 1]
            available_slots.append((end_hour, end_minute, next_start_hour, next_start_minute))
        return available_slots

    def display_full_schedule(self):
        print("\nYour full schedule for the week:")
        total_hours_per_day = self.allocate_study_hours()  # Calculate total hours per day

        for day, courses in total_hours_per_day.items():
            print(f"\n{day}:")
            total_hours = round(sum(round(hours * 2) / 2 for course, hours in courses))  # Round total hours to half-hour intervals
            if total_hours > 0:
                rounded_total_hours = round(total_hours * 2) / 2  # Round off to half-hour intervals
                print(f"  Total study hours: {rounded_total_hours:.2f}")
                for course, hours in courses:
                    if hours > 0:
                        rounded_hours = round(hours * 2) / 2  # Round off to half-hour intervals
                        print(f"    {course}: {rounded_hours:.2f} hours")
            else:
                print("  No study hours allocated.")
            if self.schedule[day]:
                print("  Busy Times:")
                for start_hour, start_minute, end_hour, end_minute in self.schedule[day]:
                    start_time = datetime(2024, 1, 1, start_hour, start_minute, tzinfo=self.pst)
                    end_time = datetime(2024, 1, 1, end_hour, end_minute, tzinfo=self.pst)
                    print(f"    {start_time.strftime('%I:%M %p')} - {end_time.strftime('%I:%M %p')}")
            else:
                print("  No commitments")




def main():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    chatbot = Chatbot()
    print("Hello! I'm your Study Hours Recommendation Chatbot.")
    chatbot.collect_class_data()
    if chatbot.classes:
        chatbot.update_model()
        chatbot.get_study_hours_recommendations()
        chatbot.collect_schedule()
        chatbot.suggest_study_hours()
        chatbot.display_full_schedule()

if __name__ == "__main__":
    main()
