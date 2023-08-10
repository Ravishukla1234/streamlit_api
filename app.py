import streamlit as st
from text2sql import *
from tabulate import tabulate


# Define the function to generate a response
def generate_response(input_text):
    response, sql_query  =  run_query(input_text)
    #print(response)
    return response, sql_query

# Create the Streamlit app
def main():
    st.title("Generative AI + Modern Data Architecture")
    
    # Input text 
    input_text = st.text_input("Enter your message")
    
    if st.button("Generate Response"):
        if input_text.strip() != "":
            # Generate response
            response, sql_query = generate_response(input_text)
            result = str(response)[2:-3]
            if "(" in result:
                
                result_list = []
                print(result)

                result = f"({result}/)"
                
                s = ""
                for x in result:
                    if x == "(" or x == "\'":
                        continue
                    elif x == ")":
                        if s[0] == ",":
                            s = s[2:]
                        print(s)
                        result_list.append(s.split(","))
                        s = ""
                    else:
                        s = s + x
                    
                    result = tabulate(result_list)
                    print(result)
                # st.table(list(result))
                
            st.text(f"Result - {result}")
               

            with st.expander("View the SQL Query generated by Gen AI engine"):
                 st.write(sql_query)
        else:
            st.warning("Please enter a Query..")

# Run the app
if __name__ == "__main__":
    main()
