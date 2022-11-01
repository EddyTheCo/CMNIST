#include"custom-datasets/cmnist.hpp"
namespace custom_models{
	namespace datasets{

uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}
CMNIST::CMNIST(const std::string& root, Mode mode):is_train_(mode==Mode::kTrain)
{
	if(mode==Mode::kTrain)
	{
		std::ifstream file_image((root+"/train-images-idx3-ubyte").c_str(), std::ios::in | std::ios::binary);
		std::ifstream file_target((root+"/train-labels-idx1-ubyte").c_str(), std::ios::in | std::ios::binary);
		int32_t M,Nitemsi,Nrowsi,Ncolumnsi;
		file_image.read(reinterpret_cast<char *>(&M), sizeof(M));
		file_image.read(reinterpret_cast<char *>(&Nitemsi), sizeof(Nitemsi));
		file_image.read(reinterpret_cast<char *>(&Nrowsi), sizeof(Nrowsi));
		file_image.read(reinterpret_cast<char *>(&Ncolumnsi), sizeof(Ncolumnsi));

		Nitems = swap_endian(Nitemsi);
		int32_t Ncolumns = swap_endian(Ncolumnsi);
		int32_t Nrows= swap_endian(Nrowsi);
		std::cout<<"Reading mnist in "<<root<<std::endl;
		std::cout<<"Nitems:"<<Nitems<<" Ncolumns:"<<Ncolumns<<" Nrows:"<<Nrows<<std::endl;
        images_= torch::empty({Nitems,1,Ncolumns,Nrows},torch::dtype(torch::kUInt8));
        auto at= images_.accessor<uint8_t,4> ();
		for(auto i=0;i<Nitems;i++)
		{
			for(auto j=0;j<Ncolumns;j++)
			{
				for(auto k=0;k<Nrows;k++)
				{
                    file_image.read(reinterpret_cast<char *>(&at[i][0][j][k]), sizeof(at[i][0][j][k]));
				}
			}
		}
        images_=images_.to(at::get_default_dtype()).div_(255);
		file_target.seekg(8);
		targets_=torch::empty({Nitems},torch::dtype(torch::kUInt8));
		auto at2= targets_.accessor<uint8_t,1> ();
		for(auto i=0;i<Nitems;i++)
		{
			file_target.read(reinterpret_cast<char *>(&at2[i]), sizeof(at2[i]));
		}

	}
	else
	{
		std::ifstream file_image((root+"/t10k-images-idx3-ubyte").c_str(), std::ios::in | std::ios::binary);
		std::ifstream file_target((root+"/t10k-labels-idx1-ubyte").c_str(), std::ios::in | std::ios::binary);
		int32_t M,Nitemsi,Nrowsi,Ncolumnsi;
		file_image.read(reinterpret_cast<char *>(&M), sizeof(M));
		file_image.read(reinterpret_cast<char *>(&Nitemsi), sizeof(Nitemsi));
		file_image.read(reinterpret_cast<char *>(&Nrowsi), sizeof(Nrowsi));
		file_image.read(reinterpret_cast<char *>(&Ncolumnsi), sizeof(Ncolumnsi));

		Nitems = swap_endian(Nitemsi);
		int32_t Ncolumns = swap_endian(Ncolumnsi);
		int32_t Nrows= swap_endian(Nrowsi);

        images_= torch::empty({Nitems,1,Ncolumns,Nrows},torch::dtype(torch::kUInt8));
        auto at= images_.accessor<uint8_t,4> ();
		for(auto i=0;i<Nitems;i++)
		{
			for(auto j=0;j<Ncolumns;j++)
			{
				for(auto k=0;k<Nrows;k++)
				{
                    file_image.read(reinterpret_cast<char *>(&at[i][0][j][k]), sizeof(at[i][0][j][k]));
				}
			}
		}
        images_=images_.to(at::get_default_dtype()).div_(255);
		file_target.seekg(8);
		targets_=torch::empty({Nitems},torch::dtype(torch::kUInt8));
		auto at2= targets_.accessor<uint8_t,1> ();
		for(auto i=0;i<Nitems;i++)
		{
			file_target.read(reinterpret_cast<char *>(&at2[i]), sizeof(at2[i]));
		}

	}

}
torch::data::Example<> CMNIST::get(size_t index)
{
	return {images_[index],targets_[index]};
}
c10::optional<size_t> CMNIST::size() const
{
	return Nitems;
}

	}//dataset namespace
}//custom_module namespace
