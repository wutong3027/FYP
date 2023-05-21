import io
import fitz
from ArticleSummarization.classes.naiveBayes import NaiveBayes
from ArticleSummarization.classes.neuralNetwork import NeuralNetwork
from ArticleSummarization.classes.decisionTree import DecisionTree

class User:
    
    def upload_file(self, pdf_file):
        text = ""
        if pdf_file is not None:
            with io.BytesIO(pdf_file.read()) as pdf_buffer:
                pdf_doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")
                for page_num in range(pdf_doc.page_count):
                    page = pdf_doc.load_page(page_num)
                    text += page.get_text()
                             
        return text
    
    # Generate summaries
    def summarize(self, text, mode):
        summary = ""
        if mode is None:
            mode = 'naive_bayes'
        # Define section keywords
        section_keywords = {
            'introduction': ['introduction', 'background', 'motivation'],
            'related work': ['related work', 'previous work', 'literature review','related works', 'review', 'reviews'],
            'methodology': ['methodology', 'method', 'approach'],
            'results': ['results', 'findings', 'experiment'],
            'conclusion': ['discussion', 'conclusion', 'implications', 'future work', 'future works'],
        }
        # Split text into sections based on section headers
        sections = {}
        current_section = None
        for line in text.splitlines():
            # Check if line matches any of the section headers
            matched_header = None
            for header, keywords in section_keywords.items():
                if any(keyword.lower() in line.lower() for keyword in keywords) and len(line.split()) <= 5:
                    matched_header = header
                    break

            if matched_header is not None:
                # If so, start a new section
                current_section = matched_header
                sections[current_section] = ""
            # Otherwise, add the line to the current section (if one exists)
            elif current_section is not None:
                sections[current_section] += line.strip() + " "
            # If the line doesn't match any section header and a section is currently open,
            # add the line to the current section. Otherwise, create a new "Miscellaneous" section.
            else:
                if "Overall" not in sections:
                    sections["Overall"] = ""
                sections["Overall"] += line.strip() + " "


        section_summaries = {}
        for text in sections:
            # Select classifier based on mode and fit classifier to data
            if mode == 'naive_bayes':
                section_summaries[text] = NaiveBayes().NB_generate_summary(sections[text])
            elif mode == 'neural_network':
                section_summaries[text] = NeuralNetwork().NN_generate_summary(sections[text])
            elif mode == 'decision_tree':
                section_summaries[text] = DecisionTree().DT_generate_summary(sections[text])
            else:
                section_summaries[text] = NaiveBayes().NB_generate_summary(sections[text])
            
        # Join section summaries with line breaks
        summary = '\n'.join([f"{section.upper()}\n{section_summaries[section]}\n" for section in section_summaries])
        print(summary)
        return summary
        
