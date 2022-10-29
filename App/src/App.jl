module App

# using Flux: DataLoader
# using CSV, Images, FileIO, DataFrames, FastAI, FastVision, FastAI.Flux, ArgParse, Metalhead
using CSV, DataFrames, Random, ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--debug_csv"
            help = "Bypass predict and write a dummy submission.csv"
            action = :store_true
        "--test_data", "-t"
            help = "The directory of the test data including test.csv and test_images/"
            arg_type = String
            default = "./"
        "--model", "-m"
            help = "The location of the inference model"
            arg_type = String
            default = "./model.jld2"
    end
    return parse_args(s)
end

function predict(id, args)
    image = Images.load(joinpath(args["test_data"], "test_images", string(id) * ".tiff"))

	#TILEIMAGE
    if args["no_tiles"]
        batch = image
    else
	    batch = tileimage(image; stepsize=args["tilesize"])
    end
	#RUNMODEL
    if args["no_tiles"]
        preds = runmodel(batch, args)
    else
        preds = predict_by_part(batch, args)
    end
	#COMPOSEMASK
    if args["no_tiles"]
        mask = preds
    else
        mask = composemask(preds, image, args)
    end
	#RLE
	return encode_rle(mask)
end

function generate_submission(df::DataFrame, args)
    df_subm = DataFrame()
    if args["debug_csv"]
        for row in 1:nrow(df)
            guess = string(round(rand(25.1:130.0); digits=3))
            df_row = DataFrame(":seq_id" => df[row, :seq_id], "tm" => guess)
            df_subm = vcat(df_subm, df_row)
        end
    else
        for row in 1:nrow(df)
            @info "Predicting $(df[row, :id])"
            df_row = DataFrame("id" => df[row, :id], "rle" => predict(df[row, :id], args))
            df_subm = vcat(df_subm, df_row)
        end
    end
    return df_subm
end

function write_submission(df::DataFrame)
    CSV.write("submission.csv", df)
end

function real_main()
    args = parse_commandline()
    df = DataFrame(CSV.File(joinpath(args["test_data"], "test.csv")))
    df_subm = generate_submission(df, args)
    write_submission(df_subm)
end

function julia_main()::Cint
  # do something based on ARGS?
  try
      real_main()
  catch
      Base.invokelatest(Base.display_error, Base.catch_stack())
      return 1
  end
  return 0 # if things finished successfully
end

end # module
