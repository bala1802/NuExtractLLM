import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def predict_NuExtract(model,tokenizer,text, schema,example = ["","",""]):
    schema = json.dumps(json.loads(schema), indent=4)
    input_llm =  "<|input|>\n### Template:\n" +  schema + "\n"
    for i in example:
      if i != "":
          input_llm += "### Example:\n"+ json.dumps(json.loads(i), indent=4)+"\n"
    
    input_llm +=  "### Text:\n"+text +"\n<|output|>\n"
    input_ids = tokenizer(input_llm, return_tensors="pt",truncation = True, max_length = 4000).to("cuda")

    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    return output.split("<|output|>")[1].split("<|end-output|>")[0]

model = AutoModelForCausalLM.from_pretrained("numind/NuExtract", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract", trust_remote_code=True)

model.to("cuda")

text = """
Sachin Tendulkar, often referred to as the “Little Master” or “Master Blaster,” made his debut for the Indian cricket team on November 15, 1989, against Pakistan in Karachi. Over his illustrious career, Tendulkar played 200 Test matches and 463 
One Day Internationals (ODIs), becoming one of the most prolific batsmen in the history of cricket. His highest score in Tests is 248 not out, achieved against Bangladesh in 2004, while his highest ODI score is a remarkable 200 not out against South Africa 
in 2010, making him the first player to score a double century in ODIs. Tendulkar has scored 51 centuries and 68 fifties in Test matches, along with 49 centuries and 96 fifties in ODIs. Under his presence, India won numerous matches, although his contributions 
went beyond personal records, inspiring countless victories and elevating the global profile of Indian cricket. Sourav Ganguly, fondly known as “Dada,” made his debut for India in an ODI against the West Indies on January 11, 1992, and his Test debut on 
June 20, 1996, against England at Lord’s, where he scored a magnificent 131. Ganguly played 113 Tests and 311 ODIs in his career, establishing himself as a key player and later, a successful captain. His highest score in Test cricket is 239 against Pakistan 
in 2007, while in ODIs, his top score is 183 against Sri Lanka in the 1999 Cricket World Cup. Ganguly scored 16 centuries and 35 fifties in Test matches, along with 22 centuries and 72 fifties in ODIs. As captain, he led India to several historic wins, 
including the famous 2002 NatWest Series victory and the 2003 Cricket World Cup final. Ganguly’s leadership and batting prowess significantly contributed to the Indian team’s success during his tenure.
"""

schema = """[
	{
	"Name":"",
	"Debut":{
		"Against":"",
		"When":"",
		"Place":""
		},
	"ODI":{
		"Total Number of Matches":"",
		"Number of Centuries":"",
		"Number of 50s":""
		},
	"Test":{
		"Total Number of Matches":"",
		"Number of Centuries":"",
		"Number of 50s":""
		}
	
	}
]"""

prediction = predict_NuExtract(model,tokenizer,text, schema,example = ["","",""])
print(prediction)

"""
Output:

[
    {
        "Name": "Sachin Tendulkar",
        "Debut": {
            "Against": "Pakistan",
            "When": "November 15, 1989",
            "Place": "Karachi"
        },
        "ODI": {
            "Total Number of Matches": "463",
            "Number of Centuries": "49",
            "Number of 50s": "96"
        },
        "Test": {
            "Total Number of Matches": "200",
            "Number of Centuries": "51",
            "Number of 50s": "68"
        }
    },
    {
        "Name": "Sourav Ganguly",
        "Debut": {
            "Against": "West Indies",
            "When": "January 11, 1992",
            "Place": ""
        },
        "ODI": {
            "Total Number of Matches": "311",
            "Number of Centuries": "22",
            "Number of 50s": "72"
        },
        "Test": {
            "Total Number of Matches": "113",
            "Number of Centuries": "16",
            "Number of 50s": "35"
        }
    }
]

"""