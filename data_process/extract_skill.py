from ojd_daps_skills.pipeline.extract_skills.extract_skills import ExtractSkills

def extract_job_skills(job_description):
    es = ExtractSkills(config_name="extract_skills_toy", local=True)
    es.load()
    
    job_adverts = [job_description.lower()]
    job_skills_matched = es.get_skills(job_adverts)
    
    return job_skills_matched


job_description = """
    The job involves Excel skills. You will also need good presentation skills
"""

job_skills = extract_job_skills(job_description)
print(job_skills[0]['SKILL'])