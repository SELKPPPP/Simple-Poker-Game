.PHONY: deploy-db delete-db deploy-lambda delete-lambda deploy-frontend delete-frontend

deploy-db:
	aws cloudformation create-stack \
		--stack-name deploy-db \
		--template-body file://deploy-db.yaml \
		--capabilities CAPABILITY_NAMED_IAM \
		--output json

delete-db:
	aws cloudformation delete-stack \
		--stack-name deploy-db \
		--output json

deploy-lambda:
	aws cloudformation create-stack \
		--stack-name deploy-lambda \
		--template-body file://deploy-lambda.yaml \
		--capabilities CAPABILITY_NAMED_IAM \
		--output json

delete-lambda:
	aws cloudformation delete-stack \
		--stack-name deploy-lambda \
		--output json

deploy-frontend:
	aws cloudformation create-stack \
		--stack-name deploy-frontend \
		--template-body file://deploy-frontend.yaml \
		--output json

delete-frontend:
	aws cloudformation delete-stack \
		--stack-name deploy-frontend \
		--output json



