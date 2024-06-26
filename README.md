# NuExtractLLM
Using NuExtractLLM, explored the extraction of entities from the unstructured text dataset

	- NuExtract from NuMind is an open-source lightweight text-to-JSON LLM
	- It allows us to extract arbitrarily complex information from text and turns it into structured data
	- Can be directly used in zero-shot setting or fine-tuned to solve a specific extraction problem

Currently, NuExtract is available in three different versions

	- NuExtract-tiny (0.5B)
	- NuExtract (3.8B)
	- NuExtract-large (7B)

	1. Structured Extraction is a highly general and adaptable information extraction task.
	2. Its objective is to pull various types of information from a document, such as:
		a. Entities
		b. Quantities
		c. Dates
	3. It seeks to determine the hierarchical relationships between the extracted details.
	4. The information is organized in a tree structure.
	5. This tree typically follows a template (schema) to make it easy to interpret.
	6. The structured information can then be:
		a. Used to populate a database
		b. Directly utilized for automatic actions.
