import re

def extract_py(code_content): 
    import_statements = re.findall(r"(?:import\s+(\w+))|(?:from\s+(\w+)\s+import\s+(\(?\w+(?:,\s*\w+)*(\)?)|\*))", code_content)

    api_calls = []
    for import_statement in import_statements:
        module1, module2, import_items, _ = import_statement
        if module1:
            api_calls.append(module1)
    
        elif module2:
            if import_items == '*':
                
                api_calls.append(module2)
            else:
                
                import_items = import_items.strip('()')
                
                import_items_list = import_items.split(',')
                
                import_items_list = [item.strip() for item in import_items_list]
                
                api_calls.extend(f"{module2}.{import_item}" for import_item in import_items_list)
    return api_calls


def extract_java(code_content): 

    pattern = r"import\s+(static\s+)?([\w.]+)"
    matches = re.findall(pattern, code_content)

    
    api_calls = [match[1].strip('.') for match in matches]
    return api_calls


def extract_c(code_content): 
    pattern = r"#include\s*(<[^>]+>|\"[^\"]+\")"
    matches = re.findall(pattern, code_content)

    api_calls = [match.strip('<>"') for match in matches]
    return api_calls

def extract_js_type(code_content): 
    pattern = r"(import\s+(?:\{\s*[\w,\s]+\s*\}\s+from\s+)?'(.+?)')|(require\(['\"](.+?)['\"]\))"
    matches = re.findall(pattern, code_content)

    api_calls = []
    for match in matches:
        if match[0]:  
            api_calls.append(match[1])
        elif match[2]:  
            api_calls.append(match[3])
    return api_calls

def extract_PHP(code_content): 

    pattern = r"\b(\w+)\("
    matches = re.findall(pattern, code_content)

    api_calls = matches
    return api_calls

def extract_Csharp(code_content): 
    
    namespace_pattern = r"using\s+([\w\.]+)\s*;"

    api_calls = re.findall(namespace_pattern, code_content)

    return api_calls

def extract_go(code_content): 
    
    single_line_pattern = r'import\s+(?:"([^"]+)"|`([^`]+)`)'

    multi_line_pattern = r'import\s*\(([^)]+)\)'

    single_line_matches = re.findall(single_line_pattern, code_content)

    multi_line_matches = re.findall(multi_line_pattern, code_content)

    api_calls = []
    
    for match in single_line_matches:
        
        api = [s.strip() for s in match if s.strip()][0]
        api_calls.append(api)

    for match in multi_line_matches:
        imports = [api.strip().strip('"').strip("'") for api in match.split('\n') if api.strip()]
        api_calls.extend(imports)
    return api_calls

def extract_css(code_content): 
    
    pattern = r'@import\s+url\(([^)]+)\)|url\(([^)]+)\)'

    matches = re.findall(pattern, code_content)

    api_calls = []

    for match in matches:
        
        api = [s.strip().strip("'").strip('"') for s in match if s.strip()][0]
        api_calls.append(api)
    return api_calls

def extract_scala(code_content): 

    pattern = r'import\s+([^\s{]+(?:\.[^\s{]+)*)(?:\.{|\s+{|\s+=>)?'

    matches = re.findall(pattern, code_content)
    api_calls = []
    for match in matches:
        api = match.strip('.')
        api_calls.append(api)
    return  api_calls


def extract_perl(code_content): 
    pattern = r'(?:use|require)\s+([^\s;]+)'

    matches = re.findall(pattern, code_content)
    api_calls = []
    
    for match in matches:
        
        api = match.strip().strip('.')
        api_calls.append(api)
    return api_calls

def extract_fortran(code_content): 
    pattern = r'use\s+(\w+)(?:\s*,\s*only\s*:\s*(\w+(?:\s*,\s*\w+)*))?'

    matches = re.findall(pattern, code_content, re.IGNORECASE)
    api_calls = []
    for match in matches:
        module_name = match[0]
        subroutine_list = match[1]
        api_calls.append(module_name)
    return api_calls


def extract_cmake(code_content): 
    pattern = r'include\(([a-zA-Z0-9_\.]+)\)'

    matches = re.findall(pattern, code_content)
    api_calls = []
    for match in matches:
        api_calls.append(match)
    return api_calls

def extract_lua(code_content): 
    pattern = r'require\(["\']([a-zA-Z0-9_\.]+)["\']\)'
    matches = re.findall(pattern, lua_code)
    api_calls = []
    for match in matches:
        api_calls.append(match)
    return api_calls

def extract_rust(code_content): 
    pattern = r'use\s+([a-zA-Z0-9_\.:]+)'
    matches = re.findall(pattern, rust_code)
    api_calls = []
    for match in matches:
        api_calls.append(match)
    return api_calls

def extract_julia(code_content): 
    pattern = r'(using|import|require)(?:\s+Base\.)?\s*([a-zA-Z0-9_\.]+|\("[^"]*"\))'

    matches = re.finditer(pattern, julia_code)
    api_calls = []
    for match in matches:
        keyword = match.group(1)
        module_name = match.group(2)
        api_calls.append(module_name.strip('(').strip(')').strip('"').strip("'").strip())

    return api_calls