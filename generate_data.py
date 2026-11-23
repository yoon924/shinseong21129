import csv
import random

random.seed(42)

TOTAL_APPLICANTS = 500
TARGET_ACCEPTED = 120

applicants = []

genders = ['Male'] * 250 + ['Female'] * 250
random.shuffle(genders)

races = ['Asian'] * 200 + ['Caucasian'] * 150 + ['Hispanic'] * 80 + ['Black'] * 70
random.shuffle(races)

age_groups = ['20s'] * 230 + ['30s'] * 200 + ['40s+'] * 70
random.shuffle(age_groups)

regions = ['Seoul/Metropolitan'] * 270 + ['Regional'] * 230
random.shuffle(regions)

for i in range(1, TOTAL_APPLICANTS + 1):
    app_id = f"APP-{i:03d}"
    gender = genders[i - 1]
    race = races[i - 1]
    age = age_groups[i - 1]
    region = regions[i - 1]

    # 1. 현실적 연령대별 경력 및 프로젝트 수
    if age == '20s':
        career_years = random.randint(0, 3)
        projects_count = random.randint(1, 5)
    elif age == '30s':
        career_years = random.randint(3, 8)
        projects_count = random.randint(2, 7)
    else:  # 40s+
        career_years = random.randint(8, 15)
        projects_count = random.randint(3, 8)

    # 2. 현실적 신체 점수 (남성 평균 78점, 여성 평균 66점 생물학적 체력 차이 반영)
    if gender == 'Male':
        physical_score = int(min(100, max(45, random.gauss(78, 9))))
    else:
        physical_score = int(min(100, max(40, random.gauss(66, 9))))

    # 3. 코딩 및 인성 점수 (성별/지역과 무관하게 독립 분포)
    coding_score = int(min(100, max(35, random.gauss(70, 13))))
    personality_score = int(min(100, max(40, random.gauss(76, 11))))

    # 순수 직무 역량 지수
    career_norm = min(career_years, 8) * 1.8
    tech_index = round(
        (coding_score * 0.45) + (personality_score * 0.25) + (physical_score * 0.10) +
        (projects_count * 3.5) + career_norm, 2
    )

    applicants.append({
        'Applicant_ID': app_id,
        'Gender': gender,
        'Race': race,
        'Age_Group': age,
        'Region': region,
        'Coding_Score': coding_score,
        'Projects_Count': projects_count,
        'Career_Years': career_years,
        'Personality_Score': personality_score,
        'Physical_Score': physical_score,
        'Technical_Index': tech_index
    })

# 1. sample_applicants.csv 저장
applicants.sort(key=lambda x: int(x['Applicant_ID'].split('-')[1]))
with open('sample_applicants.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Applicant_ID', 'Gender', 'Race', 'Age_Group', 'Region',
        'Coding_Score', 'Projects_Count', 'Career_Years', 'Personality_Score', 'Physical_Score'
    ])
    for a in applicants:
        writer.writerow([
            a['Applicant_ID'], a['Gender'], a['Race'], a['Age_Group'], a['Region'],
            a['Coding_Score'], a['Projects_Count'], a['Career_Years'],
            a['Personality_Score'], a['Physical_Score']
        ])

# -------------------------------------------------------------
# 2. sample_accepted_biased.csv (다양한 속성에 걸친 현실적 복합 편향 데이터)
# -------------------------------------------------------------
for a in applicants:
    b_score = a['Technical_Index']
    # 성별 우대(+11점)
    if a['Gender'] == 'Male':
        b_score += 11.0
    # 수도권 우대(+10점)
    if a['Region'] == 'Seoul/Metropolitan':
        b_score += 10.0
    # 40대+ 과도우대(+9점)
    if a['Age_Group'] == '40s+':
        b_score += 9.0
    # 단순 신체 점수 과다 우대(+6점, 코딩 실력 떨어져도 체력으로 역전 합격 유발)
    if a['Physical_Score'] >= 82:
        b_score += 6.0
    b_score += random.uniform(-2.5, 2.5)
    a['_biased_score'] = b_score

applicants.sort(key=lambda x: x['_biased_score'], reverse=True)
accepted_biased = applicants[:TARGET_ACCEPTED]
accepted_biased_ids = sorted([a['Applicant_ID'] for a in accepted_biased], key=lambda x: int(x.split('-')[1]))

with open('sample_accepted_biased.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Applicant_ID'])
    for app_id in accepted_biased_ids:
        writer.writerow([app_id])

with open('sample_accepted.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Applicant_ID'])
    for app_id in accepted_biased_ids:
        writer.writerow([app_id])

# -------------------------------------------------------------
# 3. sample_accepted_fair.csv (4대 분야 모두 완벽하게 공정한 데이터)
# -------------------------------------------------------------
fair_accepted_set = set()
applicants.sort(key=lambda x: x['Technical_Index'] + random.uniform(-1.0, 1.0), reverse=True)

from collections import defaultdict
counts = {
    'Male': 0, 'Female': 0,
    'Seoul/Metropolitan': 0, 'Regional': 0,
    '20s': 0, '30s': 0, '40s+': 0,
    'Asian': 0, 'Caucasian': 0, 'Hispanic': 0, 'Black': 0
}
targets = {
    'Male': 60, 'Female': 60,
    'Seoul/Metropolitan': 65, 'Regional': 55,
    '20s': 55, '30s': 48, '40s+': 17,
    'Asian': 48, 'Caucasian': 36, 'Hispanic': 19, 'Black': 17
}

for a in applicants:
    g, reg, age, race = a['Gender'], a['Region'], a['Age_Group'], a['Race']
    if (counts[g] < targets[g] and 
        counts[reg] < targets[reg] and 
        counts[age] < targets[age] and 
        counts[race] < targets[race] and 
        len(fair_accepted_set) < TARGET_ACCEPTED):
        
        fair_accepted_set.add(a['Applicant_ID'])
        counts[g] += 1
        counts[reg] += 1
        counts[age] += 1
        counts[race] += 1

for a in applicants:
    if len(fair_accepted_set) >= TARGET_ACCEPTED:
        break
    if a['Applicant_ID'] not in fair_accepted_set:
        g, reg = a['Gender'], a['Region']
        if counts[g] <= targets[g] + 2 and counts[reg] <= targets[reg] + 2:
            fair_accepted_set.add(a['Applicant_ID'])
            counts[g] += 1
            counts[reg] += 1

for a in applicants:
    if len(fair_accepted_set) >= TARGET_ACCEPTED:
        break
    fair_accepted_set.add(a['Applicant_ID'])

accepted_fair_ids = sorted(list(fair_accepted_set), key=lambda x: int(x.split('-')[1]))

with open('sample_accepted_fair.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Applicant_ID'])
    for app_id in accepted_fair_ids:
        writer.writerow([app_id])

print("CSV files generated successfully.")
